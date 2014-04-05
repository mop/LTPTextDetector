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
#include <text_detector/SoftPPWordSplitter.h>

#include <opencv2/highgui/highgui.hpp>

namespace TextDetector {

SoftPPWordSplitter::SoftPPWordSplitter(bool allow_single_letters, bool verbose)
: _allow_single_letters(allow_single_letters), _verbose(verbose) 
{
}

std::vector<cv::Rect> 
SoftPPWordSplitter::split(const CCGroup &grp)
{
    cv::Rect bb = grp.get_rect();
    // generate the projection profile sums
    cv::Mat sums(1, bb.width, CV_32FC1, cv::Scalar(0));
    ProjectionProfileComputer pp_computer(cv::Size(bb.width, 1), bb.x);
    for (int i = 0; i < grp.ccs.size(); i++) {
        sums = pp_computer.compute(grp.ccs[i].pixels, sums);
    }

    int threshold = pp_computer.compute_threshold(sums);
    if (_verbose) {
        std::cout << "Projection Profile Threshold: " << threshold << std::endl;
    }
    cv::Mat gaps = sums < threshold;

    // now shrink each bounding rect on the border with the gaps matrix
    std::vector<cv::Rect> original_rects(grp.ccs.size());
    std::transform(
        grp.ccs.begin(), grp.ccs.end(), 
        original_rects.begin(), 
        [](const CC &cc) -> cv::Rect { return cc.rect; });
    std::sort(
        original_rects.begin(),
        original_rects.end(), 
        [](const cv::Rect &a, const cv::Rect &b) -> bool { return a.x < b.x; });
    RectShrinker shrinker(0.10, bb.x);
    std::vector<cv::Rect> shrinked_rects(shrinker.shrink(original_rects, gaps));
    
    //cv::Mat img(grp.get_image());
    //cv::imshow("RECTS-wo-rects", img);
    //cv::waitKey(0);
    //for (cv::Rect r : shrinked_rects) {
    //    cv::rectangle(img, r.tl(), r.br(), cv::Scalar(128));
    //}
    //cv::imshow("RECTS", img);
    //cv::waitKey(0);

    std::vector<bool> collide(bb.width, false);
    for (int i = 0; i < shrinked_rects.size(); i++) {
        for (int j = shrinked_rects[i].x; j < shrinked_rects[i].x + shrinked_rects[i].width; j++) {
            collide[j-bb.x] = true;
        }
    }

    //std::vector<bool> collide(bb.width, false);
    //for (int i = 0; i < ccs.size(); i++) {
    //    for (int j = ccs[i].rect.x; j < ccs[i].rect.x + ccs[i].rect.width; j++) {
    //        collide[j-bb.x] = true;
    //    }
    //}

    std::vector<float> heights(grp.ccs.size(), 0.0);
    std::transform(
        grp.ccs.begin(),
        grp.ccs.end(), 
        heights.begin(),
        [] (const CC &c) -> float { return c.rect.height; });
    float mean_height = cv::sum(heights)[0] / heights.size();

    // Now find the rects from this binary mask.
    // This merges overlapping/touching CCs into a single component
    std::vector<cv::Rect> rects;
    cv::Rect last_rect(bb.x, bb.y, 1, bb.height);
    
    for (int i = 0; i < collide.size(); i++) {
        if (collide[i]) {
            last_rect.width += 1;
        } else {
            if (last_rect.width > 0) {
                rects.push_back(last_rect);
            }
            last_rect = cv::Rect(bb.x + i, bb.y, 0, bb.height);
        }
    }
    if (last_rect.width > 0) {
        rects.push_back(last_rect);
    }

    if (_verbose)
        std::cout << "#Rects: " << rects.size() << std::endl;

    if (rects.size() <= 2) {
        std::vector<cv::Rect> result;
        result.push_back(bb);
        return result;
    }

    // find the dists
    std::vector<float> dists;
    for (int i = 1; i < rects.size(); i++) {
        dists.push_back(rects[i].tl().x - rects[i-1].br().x);
    }

    //  kmeans
    cv::Mat dist_mat(dists.size(), 1, CV_32FC1);
    for (size_t i = 0; i < dists.size(); i++) {
        dist_mat.at<float>(i,0) = dists[i];
    }
    cv::Mat centers;
    cv::Mat labels;//(dists.size(),1, CV_32SC1, cv::Scalar(0));
    /*
    float min = *std::min_element(dists.begin(), dists.end());
    float max = *std::max_element(dists.begin(), dists.end());
    for (size_t i = 0; i < dists.size(); i++) {
        labels.at<int>(i,0) = std::abs(dists[i] - min) < std::abs(dists[i] - max) ? 0 : 1;
    }
    */

    if (_verbose) 
        std::cout << dist_mat << std::endl;
    kmeans(dist_mat, 2, labels, cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, .01), 5, cv::KMEANS_PP_CENTERS, centers);

    if (_verbose)
        std::cout << centers << std::endl;

    std::vector<float> cpy(dists);
    std::sort(cpy.begin(), cpy.end());
    float median = cpy[cpy.size() / 2];
    if (cpy.size() % 2 == 0) {
        median = cpy[cpy.size() / 2] + cpy[cpy.size() / 2 - 1];
        median = median / 2.0f;
    }
    float medval = median;

    float height = std::abs(centers.at<float>(0,0) - centers.at<float>(1,0)) / mean_height;
    median = std::abs(centers.at<float>(0,0) - centers.at<float>(1,0)) / (median + 1e-10);
    if (_verbose) {
        std::cout << dists.size() << " " << medval << " " << median << " " << height << std::endl;
    }
    // liblinear: 92% ACC: (10-F)
    // ./train -v 10 -B 1 -w1 2 -c 100 dists_cleaned.dat   
    // do we have a single cluster?!
    //if (dists.size() > 3 && median * 0.84320891 + height * 0.3127415 < 1.23270849 ||
    //    dists.size() <= 3 && height < 0.43413942) {
    if (median * 0.33974138 + height * 0.47850904 < 0.56307525) {
        std::vector<cv::Rect> result;
        result.push_back(bb);
        return result;
    }

    // get the index of the smallest center
    int small_center = centers.at<float>(0,0) < centers.at<float>(1,0) ? 0 : 1;

    // count the distance to cluster assignments
    int cnt[2] = {0,0};
    for (int i = 0; i < labels.rows; i++) {
        cnt[labels.at<int>(i,0)]++;
    }
    // we have more word gaps than letter gaps -> don't split!
    if (cnt[small_center] < cnt[1-small_center]) {
        std::vector<cv::Rect> result;
        result.push_back(bb);
        return result;
    }

    // start from left to right and iteratively merge rects if the
    // distance between them is clustered into the smaller center
    last_rect = rects[0];
    std::vector<cv::Rect> word_candidates;
    for (int i = 1; i < rects.size(); i++) {
        if (_allow_single_letters) {
            if (labels.at<int>(i-1,0) == small_center) {
                // extend the last rect
                last_rect = last_rect | rects[i];
            } else {
                // do not extend it!
                word_candidates.push_back(last_rect);
                last_rect = rects[i];
            }
        } else {
            if (labels.at<int>(i-1,0) == small_center) {
                // extend the last rect
                last_rect = last_rect | rects[i];
            } else if (i < labels.rows && labels.at<int>(i,0) == small_center) {
                // do not extend it!
                word_candidates.push_back(last_rect);
                last_rect = rects[i];
            } else {
                last_rect = last_rect | rects[i];
            }
        }
    }
    word_candidates.push_back(last_rect);

    // for each rect, find the original connected component rects
    std::vector<cv::Rect> words;
    for (cv::Rect candidate : word_candidates) {
        std::vector<cv::Rect> word;
        for (size_t i = 0; i < grp.ccs.size(); i++) {
            cv::Rect intersect(grp.ccs[i].rect & candidate);
            if (float (intersect.width * intersect.height) / float (grp.ccs[i].rect.width * grp.ccs[i].rect.height) >= 0.8f) {
                cv::Rect r = grp.ccs[i].rect;
                // set the text height correctly
                r.y = bb.y;
                r.height = bb.height;
                word.push_back(r);
            }
        }

        if (_verbose) {
            std::cout << "Accumulated: " << word.size() << " rects!" << std::endl;
        }
        if (word.empty()) continue;
        assert(!word.empty());
        cv::Rect r = word[0];
        for (size_t i = 1; i < word.size(); i++) {
            r = r | word[i];
        }
        words.push_back(r);
    }
    
    return words;
}

}
