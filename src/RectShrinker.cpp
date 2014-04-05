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
#include <text_detector/RectShrinker.h>

namespace TextDetector {

RectShrinker::RectShrinker(float max_fraction, int offset_x)
: _max_fraction(max_fraction), _offset(offset_x) {}

RectShrinker::~RectShrinker() {}

std::vector<cv::Rect> 
RectShrinker::shrink(
    const std::vector<cv::Rect> &rects,
    const cv::Mat &gaps) const
{
    std::vector<cv::Rect> shrinked_rects;
    shrinked_rects.reserve(rects.size());

    for (size_t i = 0; i < rects.size(); i++) {
        const cv::Rect &r = rects[i];
        cv::Rect new_rect = r;
        int max_width = _max_fraction * r.width;
        // shrink the start of the new rect
        if (i > 0) { // don't shrink the first rect on the front
            int x;
            for (x = new_rect.x; gaps.at<unsigned char>(0, x-_offset) &&
                (x-_offset) < gaps.cols && 
                (x-new_rect.x) < max_width; x++) {}
            new_rect.x = x;
            new_rect.width = new_rect.width - (x - r.x);
        }

        // shrink te end of the new rect
        if (i < rects.size() - 1) { // don't shrink the last rect on the end
            int x;
            for (x = new_rect.x + new_rect.width; 
                gaps.at<unsigned char>(0, x-_offset) > 0 && 
                (x-new_rect.x) > 0 && 
                (new_rect.x + new_rect.width - x) < max_width; x--) {}
            new_rect.width = new_rect.width - (new_rect.x + new_rect.width - x);
        }
        shrinked_rects.push_back(new_rect);
    }

    return shrinked_rects;
}

}
