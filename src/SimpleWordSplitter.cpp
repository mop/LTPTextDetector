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
#include <text_detector/SimpleWordSplitter.h>

namespace TextDetector {

SimplePPWordSplitter::SimplePPWordSplitter(bool allow_single_letters, bool verbose)
: _allow_single_letters(allow_single_letters), _verbose(verbose)
{
}


std::vector<cv::Rect> 
SimplePPWordSplitter::split(const CCGroup &grp)
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

    std::vector<float> cpy(dists);
    std::sort(cpy.begin(), cpy.end());
    float median = cpy[cpy.size() / 2];
    if (cpy.size() % 2 == 0) {
        median = cpy[cpy.size() / 2] + cpy[cpy.size() / 2 - 1];
        median = median / 2.0f;
    }


              //  [[ 0.10489361  0.63316893]]
              //      [-0.59013942]

    // start from left to right and iteratively merge rects if the
    // distance between them is dead
    last_rect = rects[0];
    std::vector<cv::Rect> words;
    for (int i = 1; i < rects.size(); i++) {
        if (_allow_single_letters) {
            if (dists[i-1] / mean_height * 0.63316893 + dists[i-1] / median * 0.10489361 <= 0.5901) {
                // extend the last rect
                last_rect = last_rect | rects[i];

            } else {
                // do not extend it!
                words.push_back(last_rect);
                last_rect = rects[i];
            }
        } else {
            if (dists[i-1] / mean_height * 0.63316893 + dists[i-1] / median * 0.10489361 <= 0.5901) {
                // extend the last rect
                last_rect = last_rect | rects[i];
            } else if (i < dists.size() && dists[i-1] / mean_height * 0.63316893 + dists[i-1] * 0.10489361 <= 0.5901) {
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
