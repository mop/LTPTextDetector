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
#include <text_detector/HardPPWordSplitter.h>

namespace TextDetector {


HardPPWordSplitter::HardPPWordSplitter(bool allow_single_letters, bool verbose)
: _allow_single_letters(allow_single_letters), _verbose(verbose)
{
}

std::vector<cv::Rect> 
HardPPWordSplitter::split(const CCGroup &grp)
{
    cv::Rect bb = grp.get_rect();
    // generate a 1D-mask where the gaps between CCs are 0 and the 
    // components are 1
    std::vector<bool> collide(bb.width, false);
    for (int i = 0; i < grp.ccs.size(); i++) {
        for (int j = grp.ccs[i].rect.x; j < grp.ccs[i].rect.x + grp.ccs[i].rect.width; j++) {
            collide[j-bb.x] = true;
        }
    }

    std::vector<float> heights(grp.ccs.size(), 0.0);
    std::transform(grp.ccs.begin(), grp.ccs.end(), heights.begin(), [] (const CC &c) -> float { return c.rect.height; });
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
    cv::Mat labels;
    if (_verbose)
        std::cout << dist_mat << std::endl;
    kmeans(dist_mat, 2, labels, cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, .01), 5, cv::KMEANS_PP_CENTERS, centers);

    std::vector<float> cpy(dists);
    std::sort(cpy.begin(), cpy.end());
    float median = cpy[cpy.size() / 2];
    if (cpy.size() % 2 == 0) {
        median = cpy[cpy.size() / 2] + cpy[cpy.size() / 2 - 1];
        median = median / 2.0f;
    }

    float height = std::abs(centers.at<float>(0,0) - centers.at<float>(1,0)) / mean_height;
    median = std::abs(centers.at<float>(0,0) - centers.at<float>(1,0)) / (median + 1e-10);
    // liblinear: 92% ACC: (10-F)
    // ./train -v 10 -B 1 -w1 2 -c 100 dists_cleaned_2.dat   
    //if (median * 0.3719757435798741 + height * 0.8523389079247736 - -0.4203646300224174 < 0) {
    //if (dists.size() > 2 && median * 0.2631996557981348 + height * 0.4972964430059804 < 0.4813098260184576 || 
    //    dists.size() <= 2 && height <= 0.2828104575163398) {
   // if (median * 0.27647134 + height * 0.54699267 < 0.87536189) {
   //

    if (ConfigurationManager::instance()->verbose()) {
        std::cout << mean_height << std::endl;
        std::cout << centers << std::endl;
        std::cout << median * 2.17066083 + height * 5.7493335 - 2.9716705 << std::endl;
    }
    //[[ 0.19387692  3.29258427]]
    //    [-1.35134204]
    //    [[ 0.19730102  3.16627143]]
    //    [-1.2701561]
    //

    if (dists.size() > 2 && median * 0.19387692 + height * 3.16627143 < 1.2701561 || 
        dists.size() <= 2 && height <= 0.297) {
    //if (median * 0.27647134 + height * 0.54699267 < 0.87536189) {
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
    std::vector<cv::Rect> words;
    for (int i = 1; i < rects.size(); i++) {
        if (_allow_single_letters) {
            if (labels.at<int>(i-1,0) == small_center) {
                // extend the last rect
                last_rect = last_rect | rects[i];
            } else {
                // do not extend it!
                words.push_back(last_rect);
                last_rect = rects[i];
            }
        } else {
            if (labels.at<int>(i-1,0) == small_center) {
                // extend the last rect
                last_rect = last_rect | rects[i];
            } else if (i < labels.rows && labels.at<int>(i,0) == small_center) {
                // do not extend it!
                words.push_back(last_rect);
                last_rect = rects[i];
            } else {
                last_rect = last_rect | rects[i];
            }
        }
    }
    words.push_back(last_rect);
    
    return words;
}

}
