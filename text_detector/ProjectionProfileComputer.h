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

#ifndef PROJECTIONPROFILECOMPUTER_H

#define PROJECTIONPROFILECOMPUTER_H

#include <opencv2/core/core.hpp>

namespace TextDetector {

/**
 *  This class is responsible for computing projection 
 *  profiles from a given pixel list.
 *  It computes thresholds for this CC
 */
class ProjectionProfileComputer 
{
public:
    /**
     *  Initializes the projection profile computer for a segment with a width of 
     *  size.width, which is offseted by offset from the x-coordinate of each point.
     */
    ProjectionProfileComputer(const cv::Size &size, int offset = 0);
    ~ProjectionProfileComputer();

    /**
     *  Computes the projection profiles from the given list of points.
     *  
     *  @param el is the list of points for which the profiles are computed.
     *  @param img is a previous computed projection profile, to which the given elment el
     *         is added.
     */
    cv::Mat compute(const std::vector<cv::Point> &el, cv::Mat img = cv::Mat()) const;

    /**
     *  Computes the 0.25 percentile threshold for the given list of points
     */
    int compute_threshold(const std::vector<cv::Point> &el, float percentile = 0.25) const;
    /**
     *  Computes the 0.25 percentile threshold for the given projection profile
     */
    int compute_threshold(const cv::Mat &sum, float percentile=0.25) const;

private:
    int percentile(const cv::Mat &hist, float p) const;

    cv::Size _size;
    int _offset;
};

}

#endif /* end of include guard: PROJECTIONPROFILECOMPUTER_H */
