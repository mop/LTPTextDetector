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

#ifndef LTPCOMPUTER_H

#define LTPCOMPUTER_H

#include <opencv2/core/core.hpp>

namespace TextDetector {

class LTPComputer 
{
public:
    LTPComputer(int nmaps = 8);
    ~LTPComputer() {}

    cv::Mat compute(const cv::Mat &rgb_image) const;
    template <class T>
    cv::Mat get_vector(const cv::Mat &lbp_map, int x, int y, int ex, int ey) const;
private:
    int _nmaps;
};

template <class T>
cv::Mat LTPComputer::get_vector(const cv::Mat &lbp_map, int x, int y, int ex, int ey) const
{    
    cv::Mat patch = lbp_map.colRange(x, ex).rowRange(y, ey);
    int w = ex - x;
    int h = ey - y;
    assert(w == 24);
    assert(h == 12);
    cv::Mat vec(1, w * h * _nmaps * 2, CV_32FC1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int c = 0; c < _nmaps * 2; c++) {
                vec.at<float>(0, c*w*h + i*w + j) = *(patch.ptr<T>(i,j) + c);
            }
        }
    }
    return vec;
}

}

#endif /* end of include guard: LTPCOMPUTER_H */
