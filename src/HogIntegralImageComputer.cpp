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
#include <text_detector/HogIntegralImageComputer.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

void HogIntegralImageComputer::set_image(const cv::Mat &image)
{
    cv::Mat mags, qangles;
    _hog.computeGradient(image, mags, qangles);
    compute_integral_image(mags, qangles);
}

cv::Mat HogIntegralImageComputer::create_hog_maps(const cv::Mat &mags, const cv::Mat &qangles)
{
    cv::Mat maps(mags.rows, mags.cols, CV_32FC(9), cv::Scalar(0.0f));
    for (int i = 0; i < mags.rows; i++) {
        for (int j = 0; j < mags.cols; j++) {
            cv::Vec2b ang = qangles.at<cv::Vec2b>(i,j);
            cv::Vec2f mag = mags.at<cv::Vec2f>(i,j);
            cv::VecHogf v(0.0f);
            v[ang[0]] += mag[0];
            v[ang[1]] += mag[1];
            v[CC_HOG_CHANS] = mag[0] + mag[1]; // gradient magnitude...
            maps.at<cv::VecHogf>(i,j) = v;
        }
    }
    return maps;
}

cv::VecHogf HogIntegralImageComputer::query_ii(const cv::Point &start, const cv::Point &end) const
{
    // sanity checks
    assert(end.y < _integral_image.rows - 1 && end.x < _integral_image.cols - 1 &&
           end.x >= 0 && end.y >= 0);
    assert(start.y < _integral_image.rows - 1 && start.x < _integral_image.cols - 1 &&
           start.x >= 0 && start.y >= 0);
    return _integral_image.at<cv::VecHogf>(end.y+1, end.x+1) - 
           _integral_image.at<cv::VecHogf>(start.y, end.x+1) -
           _integral_image.at<cv::VecHogf>(end.y+1, start.x) + 
           _integral_image.at<cv::VecHogf>(start.y, start.x);
}

static cv::VecHogf vec_max(const cv::VecHogf &v1, const cv::VecHogf &v2)
{
    cv::VecHogf result;
    for (int i = 0; i < CC_HOG_CHANS+1; i++) {
        result[i] = std::max(v1[i], v2[i]);
    }
    return result;
}

cv::Mat HogIntegralImageComputer::compute_block_features(const cv::Point &start, const cv::Point &end) const
{
    // sanity checks
    assert(end.y < _integral_image.rows-1 && end.x < _integral_image.cols-1 &&
           end.x >= 0 && end.y >= 0);
    assert(start.y < _integral_image.rows-1 && start.x < _integral_image.cols-1 &&
           start.x >= 0 && start.y >= 0);
    // We have top, middle and bottom blocks
    // The border of blocks might overlap (for simplicity reasons)
    int h = end.y - start.y + 1;

    if (h < 3) {
        //return cv::Mat::zeros(1, 8*3, CV_32FC1);
        return cv::Mat::zeros(1, CC_HOG_CHANS*3, CV_32FC1);
    }

    int h_top = h * 0.2;     // the first border
    int h_bottom = h * 0.8;  // the second border

    h_top    = std::min(h_top, h - 3);
    h_bottom = std::min(h_bottom, h - 2);

    cv::Point start1 = start;
    cv::Point end1   = cv::Point(end.x, start.y + h_top);

    cv::Point start2 = cv::Point(start.x, start.y + h_top + 1);
    cv::Point end2   = cv::Point(end.x,   start.y + h_bottom);

    cv::Point start3 = cv::Point(start.x, start.y + h_bottom + 1);
    cv::Point end3   = cv::Point(end.x,   end.y);

    cv::VecHogf v1 = vec_max(query_ii(start1, end1), cv::VecHogf(0.0f));
    cv::VecHogf v2 = vec_max(query_ii(start2, end2), cv::VecHogf(0.0f));
    cv::VecHogf v3 = vec_max(query_ii(start3, end3), cv::VecHogf(0.0f));
    cv::VecHogf v_norm = v1 + v2 + v3;

    assert(v_norm[4] + 1e-5 > 0);
    v1 = v1 / (v_norm[CC_HOG_CHANS] + 1e-5);
    v2 = v2 / (v_norm[CC_HOG_CHANS] + 1e-5);
    v3 = v3 / (v_norm[CC_HOG_CHANS] + 1e-5);

#if CC_HOG_CHANS == 4
    cv::Mat result = (cv::Mat_<float>(1,CC_HOG_CHANS*3) << 
        v1[0], v1[1], v1[2], v1[3],
        v2[0], v2[1], v2[2], v2[3],
        v3[0], v3[1], v3[2], v3[3]);
#else
    cv::Mat result = (cv::Mat_<float>(1,CC_HOG_CHANS*3) << 
        v1[0], v1[1], v1[2], v1[3], v1[4], v1[5], v1[6], v1[7],
        v2[0], v2[1], v2[2], v2[3], v2[4], v2[5], v2[6], v2[7],
        v3[0], v3[1], v3[2], v3[3], v3[4], v3[5], v3[6], v3[7]);
#endif
    cv::sqrt(result, result);
    return result;
}

void HogIntegralImageComputer::compute_integral_image(const cv::Mat &mags, const cv::Mat &qangles)
{
    cv::Mat hog_maps = create_hog_maps(mags, qangles);
    cv::integral(hog_maps, _integral_image, CV_32F);
}
