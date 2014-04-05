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
#include <text_detector/ProjectionProfileComputer.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace TextDetector {

ProjectionProfileComputer::ProjectionProfileComputer(const cv::Size &size, int offset)
: _size(size), _offset(offset) {}

ProjectionProfileComputer::~ProjectionProfileComputer() {}

cv::Mat ProjectionProfileComputer::compute(const std::vector<cv::Point> &el, cv::Mat img) const
{
    if (img.rows == 0) 
        img = cv::Mat(1, _size.width, CV_32FC1, cv::Scalar(0.0f));

    for (int i = 0; i < el.size(); i++) {
        img.at<float>(0,el[i].x - _offset) += 1.0;
    }
    return img;
}

int ProjectionProfileComputer::compute_threshold(const std::vector<cv::Point> &el, float p) const
{
    return compute_threshold(compute(el), p);
}

int ProjectionProfileComputer::compute_threshold(const cv::Mat &sums, float p) const
{
    // do some histograms on the profile-sums
    double mi, ma;
    cv::minMaxLoc(sums, &mi, &ma);
    const float range[] = {  1/255.0f, float(ma)  };
    const float* hist_range = { range };
    cv::Mat hist;
    int hist_size = 64;

    cv::calcHist(&sums, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range);

    // calculate the percentile bucket of the histogram and scale it appropriately
    return float(percentile(hist, p)) / float(hist_size) * ma;
}

int ProjectionProfileComputer::percentile(const cv::Mat &hist, float p) const
{
    int size = 0;
    for (int i = 0; i < hist.rows; i++) size += hist.at<float>(i, 0);

    float accum = 0;
    int idx = -1;
    for (int i = 0; i < hist.rows; i++) {
        accum += hist.at<float>(i,0);
        if (accum > size * p) {
            idx = i+1; break;
        }
    }

    return idx;
}

}
