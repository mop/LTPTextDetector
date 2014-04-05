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

#ifndef RECTSHRINKER_H

#define RECTSHRINKER_H

#include <opencv2/core/core.hpp>

namespace TextDetector {

/**
 *  This class is responsible for shrinking the borders of list of 
 *  rectangles with a binary mask. It is used after the projection profile
 *  computation to modify the bounding rectangles of each character.
 */
class RectShrinker 
{
public:
    RectShrinker(float max_fraction=0.10, int offset_x = 0);
    ~RectShrinker();

    std::vector<cv::Rect> shrink(const std::vector<cv::Rect> &rects, const cv::Mat &mask) const;

private:
    int _offset;
    float _max_fraction;
};

}


#endif /* end of include guard: RECTSHRINKER_H */
