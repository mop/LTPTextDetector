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
#include <text_detector/CRFRFConnectedComponentFilterer.h>

#include <text_detector/ConfigurationManager.h>
#include <numeric>
#include <boost/timer/timer.hpp>

namespace TextDetector {

CRFRFConnectedComponentFilterer::CRFRFConnectedComponentFilterer(
    const cv::Mat &train_image, 
    const cv::Mat &gradient_image,
    const dlib::graph_labeler<vector_type> &labeler, 
    std::shared_ptr<cv::RandomTrees> pairwise_1_1_tree,
    std::shared_ptr<cv::RandomTrees> pairwise_1_0_tree,
    std::shared_ptr<cv::RandomTrees> pairwise_0_0_tree
): _train_image(train_image), 
   _gradient_image(gradient_image),
   _labeler(labeler),
   _pairwise_1_1_tree(pairwise_1_1_tree),
   _pairwise_1_0_tree(pairwise_1_0_tree),
   _pairwise_0_0_tree(pairwise_0_0_tree) {}

CRFRFConnectedComponentFilterer::~CRFRFConnectedComponentFilterer() {}

std::vector<std::pair<int, std::vector<cv::Point> > >  
CRFRFConnectedComponentFilterer::filter_compontents(
    const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
    std::vector<MserElement> &all_elements,
    const std::vector<std::vector<double> > &per_classifier_probs
)
{
    graph_type graph;
    graph.set_number_of_nodes(comps.size());

    // set the data for each node

    boost::timer::cpu_timer t;
    t.start();
    for (size_t i = 0; i < comps.size(); i++) {
        node_vector_type data;
        data(0,0) = 1; // bias
        int idx = comps[i].first;
        data(1,0) = per_classifier_probs[idx][0];

        graph.node(i).data = data;
    }

    if (ConfigurationManager::instance()->verbose())
        std::cout << "Finished unary features: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

    // dists is an adjacency list where the first element in the pair is the distance to element and the 
    // second element in the pair is the index of the mser-element
    t.start();
    std::vector<std::vector<std::pair<float, int> > > dists;
    dists.resize(comps.size());

    // pairwise stuff....
    #pragma omp parallel for
    for (size_t i = 0; i < comps.size(); i++) {
        const MserElement &el1 = all_elements[comps[i].first];
        cv::Rect r1 = el1.get_bounding_rect();
        cv::Vec2f c1 = el1.get_centroid();
        for (size_t j = 0; j < comps.size(); j++) {
            if (i == j) continue;

            const MserElement &el2 = all_elements[comps[j].first];
            cv::Rect r2 = el2.get_bounding_rect();
            cv::Vec2f c2 = el2.get_centroid();
            double maxw = std::max(r1.width, r2.width);
            cv::Vec2f dist = c1 - c2;
            float d = sqrt(dist[0] * dist[0] + dist[1] * dist[1]);

            if (d < 2.0 * maxw) {
                dists[i].push_back(std::make_pair(d, j));
            }
        }
        std::sort(dists[i].begin(), dists[i].end(), [] (const std::pair<float, int> &p1, const std::pair<float, int> &p2) ->  bool { 
            return p1.first < p2.first; 
        });
    }

    if (ConfigurationManager::instance()->verbose())
        std::cout << "Finished pairwise distances in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    t.start();

    // an adjacency matrix
    cv::Mat adj(comps.size(), comps.size(), CV_8UC1, cv::Scalar(0));
    //cv::Mat probs(comps.size(), comps.size(), CV_32FC3, cv::Scalar(-1.0f, -1.0f, -1.0f));
    int size[2] = { std::max(1, int(comps.size())), std::max(1, int(comps.size())) };
    cv::SparseMat probs(2, size, CV_32FC3);

    // Since our graph lib screws up on parallel access, feature computation and graph 
    // construction is split up
    #pragma omp parallel for
    for (size_t i = 0; i < dists.size(); i++) {
        if (dists[i].empty()) continue;

        for (size_t j = 0; j < N_NEIGHBORS && j < dists[i].size(); j++) {
            int idx_i = i;
            int idx_j = dists[i][j].second;

            bool skip = false;
            #pragma omp critical
            {
            if (adj.at<unsigned char>(idx_i, idx_j) || adj.at<unsigned char>(idx_j, idx_i)) {
                skip = true;
            } else {

                adj.at<unsigned char>(idx_i, idx_j) = 1;
                adj.at<unsigned char>(idx_j, idx_i) = 1;
            }
            }
            if (skip) continue;


            cv::Mat f = all_elements[comps[idx_i].first].compute_pairwise_features(
                _train_image, _gradient_image, all_elements[comps[idx_j].first]).colRange(0,9);
            
            cv::Vec3f prob;
            prob[0] = _pairwise_1_1_tree->predict(f);
            prob[1] = _pairwise_1_0_tree->predict(f);
            prob[2] = _pairwise_0_0_tree->predict(f);
            //probs.at<cv::Vec3f>(idx_i, idx_j) = prob;
            //probs.at<cv::Vec3f>(idx_j, idx_i) = prob;
            //
            #pragma omp critical
            {
                probs.ref<cv::Vec3f>(idx_i, idx_j) = prob;
                probs.ref<cv::Vec3f>(idx_i, idx_j) = prob;
            }
        }
    }

    adj = cv::Scalar(0);
    for (size_t i = 0; i < dists.size(); i++) {
        if (dists[i].empty()) continue;
        for (size_t j = 0; j < N_NEIGHBORS && j < dists[i].size(); j++) {
            int idx_i = i;
            int idx_j = dists[i][j].second;
            //cv::Vec3f p = probs.at<cv::Vec3f>(idx_i, idx_j);
            //if (p[0] <= -1) continue;
            auto ptr = probs.find<cv::Vec3f>(idx_i, idx_j);
            if (!ptr) continue;
            if (adj.at<unsigned char>(idx_i, idx_j) || adj.at<unsigned char>(idx_j, idx_i))
                    continue;
            const cv::Vec3f p = *ptr;
            adj.at<unsigned char>(idx_i, idx_j) = 1;
            adj.at<unsigned char>(idx_j, idx_i) = 1;

            edge_vector_type data;
            data(0,0) = 1; // bias
            data(1,0) = p[0];
            data(2,0) = p[1];
            data(3,0) = p[2];

            graph.add_edge(idx_i, idx_j);
            dlib::edge(graph, idx_i, idx_j) = data;
        }
    }
    if (ConfigurationManager::instance()->verbose())
        std::cout << "Finished pairwise features in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

    adj.release();
    probs.release();
    dists.clear();
    // inference!
    std::vector<bool> labels = _labeler(graph);

    if (ConfigurationManager::instance()->verbose())
        std::cout << "Finished inference" << std::endl;

    std::vector<std::pair<int, std::vector<cv::Point> > > result; 
    result.reserve(comps.size());
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i])
            result.push_back(comps[i]);
    }

    return result;
}

}
