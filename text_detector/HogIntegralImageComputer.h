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

#ifndef HOGINTEGRALIMAGECOMPUTER_H

#define HOGINTEGRALIMAGECOMPUTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "config.h"

namespace cv {
#if CC_HOG_CHANS == 8
    typedef Vec<float, 9> VecHogf;
#else 
    typedef Vec<float, 5> VecHogf;
#endif
}
class HogIntegralImageComputer 
{
public:
    HogIntegralImageComputer(): _hog(cv::HOGDescriptor(cv::Size(12,24), cv::Size(4,4), cv::Size(4,4), cv::Size(4,4), 8)) {}
    ~HogIntegralImageComputer() {}

    void set_image(const cv::Mat &image);
    cv::VecHogf query_ii(const cv::Point &start, const cv::Point &end) const;
    cv::Mat compute_block_features(const cv::Point &start, const cv::Point &end) const;
    cv::Mat compute_feature(const cv::Point &start, const cv::Point &end) const { return compute_block_features(start, end); }

    cv::Size get_size() const { return cv::Size(_integral_image.cols - 1, _integral_image.rows - 1); }

private:
    cv::Mat create_hog_maps(const cv::Mat &mags, const cv::Mat &qangles);
    void compute_integral_image(const cv::Mat &gradients, const cv::Mat &qangles);
    cv::HOGDescriptor _hog;

    cv::Mat _integral_image;
};

#endif /* end of include guard: HOGINTEGRALIMAGECOMPUTER_H */
