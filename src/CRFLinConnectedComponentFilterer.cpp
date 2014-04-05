/**
 *  This file is part of ltp-text-detector.
 *  Copyright (C) 2013 Michael Opitz
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <text_detector/CRFLinConnectedComponentFilterer.h>

namespace TextDetector {

CRFLinConnectedComponentFilterer::CRFLinConnectedComponentFilterer(
    const cv::Mat &train_image,
    const cv::Mat &gradient_image,
    const cv::Mat &unary_features,
    const dlib::graph_labeler<vector_type> &labeler
): 
   _train_image(train_image),
   _gradient_image(gradient_image),
   _unary_features(unary_features),
   _labeler(labeler) {}

CRFLinConnectedComponentFilterer::~CRFLinConnectedComponentFilterer() {}

std::vector<std::pair<int, std::vector<cv::Point> > >
CRFLinConnectedComponentFilterer::filter_compontents(
    const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
    std::vector<MserElement> &all_elements,
    const std::vector<std::vector<double> > &per_classifier_probs
)
{
    graph_type graph;
    graph.set_number_of_nodes(comps.size());

    // set the data for each node
    for (size_t i = 0; i < comps.size(); i++) {
        node_vector_type data;
        data(0,0) = 1; // bias
        int idx = comps[i].first;
        for (int j = 0; j < _unary_features.cols; j++) {
            data(j+1,0) = _unary_features.at<float>(i,j);
        }

        graph.node(i).data = data;
    }

    // dists is an adjacency list where the first element in the pair is the distance to element and the 
    // second element in the pair is the index of the mser-element
    std::vector<std::vector<std::pair<float, int> > > dists;
    dists.resize(comps.size());

    // pairwise stuff....
    for (size_t i = 0; i < comps.size(); i++) {
        const MserElement &el1 = all_elements[comps[i].first];
        cv::Rect r1 = el1.get_bounding_rect();
        cv::Vec2f c1 = el1.get_centroid();
        for (size_t j = 0; j < comps.size(); j++) {
            if (i == j) continue;

            const MserElement &el2 = all_elements[comps[j].first];
            cv::Rect r2 = el2.get_bounding_rect();
            cv::Vec2f c2 = el2.get_centroid();
            double maxw = std::min(std::max(r1.width, r1.height), std::max(r2.width, r2.height));
            cv::Vec2f dist = c1 - c2;
            float d = sqrt(dist[0] * dist[0] + dist[1] * dist[1]);

            if (d < 2.0 * maxw) {
                dists[i].push_back(std::make_pair(d, j));
            }
        }
    }

    // an adjacency matrix
    cv::Mat adj(comps.size(), comps.size(), CV_8UC1, cv::Scalar(0));

    for (size_t i = 0; i < dists.size(); i++) {
        if (dists[i].empty()) continue;
        std::sort(dists[i].begin(), dists[i].end(), [] (const std::pair<float, int> &p1, const std::pair<float, int> &p2) ->  bool { return p1.first < p2.first; });

        for (size_t j = 0; j < N_NEIGHBORS && j < dists[i].size(); j++) {
            int idx_i = i;
            int idx_j = dists[i][j].second;

            if (adj.at<unsigned char>(idx_i, idx_j) || adj.at<unsigned char>(idx_j, idx_i)) continue;

            cv::Mat f = all_elements[comps[idx_i].first].compute_pairwise_features(
                _train_image, _gradient_image, all_elements[comps[idx_j].first]).colRange(0,9);
            edge_vector_type data;
            data(0,0) = 1; // bias
            for (int k = 0; k < f.cols; k++) {
                data(k+1,0) = f.at<float>(0,k);
            }

            graph.add_edge(idx_i, idx_j);
            dlib::edge(graph, idx_i, idx_j) = data;

            adj.at<unsigned char>(idx_i, idx_j) = 1;
            adj.at<unsigned char>(idx_j, idx_i) = 1;
        }
    }

    // inference!
    std::vector<bool> labels = _labeler(graph);

    std::vector<std::pair<int, std::vector<cv::Point> > > result; 
    result.reserve(comps.size());
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i])
            result.push_back(comps[i]);
    }

    // visualize things
    //cv::Mat img(train_image.rows, train_image.cols, CV_8UC3, cv::Scalar(0));
    //for (int i = 0; i < comps.size(); i++) {
    //    int idx = comps[i].first;
    //    all_elements[idx].set_label(labels[i]);
    //    all_elements[idx].draw(img);
    //}
    //for (int i = 0; i < comps.size(); i++) {
    //    int idx = comps[i].first;
    //    cv::Vec2f c1 = all_elements[idx].get_centroid();
    //    for (int j = 0; j < comps.size(); j++) {
    //        if (j == i) continue;

    //        if (!adj.at<unsigned char>(i,j)) continue;

    //        cv::Vec2f c2 = all_elements[comps[j].first].get_centroid();
    //        cv::line(img, cv::Point(c1[0], c1[1]), cv::Point(c2[0], c2[1]), cv::Scalar(0,0,255));
    //    }
    //}

    //cv::imshow("graph", img);
    //cv::waitKey(0);

    return result;
}

}
