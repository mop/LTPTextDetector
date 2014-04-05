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
#ifndef CCGROUP_H

#define CCGROUP_H

#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ConfigurationManager.h"
#include "ProjectionProfileComputer.h"
#include "RectShrinker.h"

namespace TextDetector {

/**
 * This class represents a connected component. It stores a pixel-list, the id into the 
 * comps vector, and an index into the feature vector/matrix.
 * From the pixel list a bounding rectangle and a centroid vector is computed.
 */
class CC
{
public:
    CC(int idx, int feature_idx, const std::vector<cv::Point> &img);

    bool can_link(const CC &rhs) const;
    float distance(const CC &rhs) const;
    void draw(cv::Mat img) const;

    int id;
    int feature_id;
    std::vector<cv::Point> pixels;
    cv::Vec2f centroid;
    cv::Rect rect;
};

/**
 *  This class manages a list of connected components. 
 */
class CCGroup 
{
public:
    CCGroup() {}
    ~CCGroup() {}

    /**
     * Draws the group
     */
    cv::Mat draw(const cv::Size &image_size, const std::vector<double> &probs, const cv::Mat &result = cv::Mat(), bool show=false) const; 

    /**
     *  Computes the single-linkage distance to the elements from the other 
     *  group `grp` using the given distance matrix.
     */
    float distance(const CCGroup &grp, const cv::Mat &distance_matrix) const;
    /**
     *  This this group with the given group grp
     */
    void link(const CCGroup &grp);
    /**
     * Returns true if the groups can be linked
     */
    bool can_link(const CCGroup &grp, const cv::Mat &distance_matrix) const;
    /**
     *  Returns the sum of the bounding box areas of each individual 
     *  connected component
     */
    float get_bounding_box_area() const;
    /**
     *  Returns the sum of the intersection bounding box areas of 
     *  each individual connected component
     */
    float get_intersection_area(const CCGroup &grp) const ;
    /**
     *  Returns the bounding rectangle of the whole group
     */
    cv::Rect get_rect() const;
    /**
     *  Returns true if the connected component with the given index 
     *  contributes 'much' to 'support' the bounding box.
     *  For a component which lies on the boundary of the box 
     *  (i.e. left or right) this returns true
     */
    bool is_significant_member(int idx) const;
    /**
     *  Returns the bounding rectangle without the connected componend 
     *  which index is indicated by `leave_out`
     */
    cv::Rect get_rect_without_element(int leave_out) const;
    /**
     *  Returns the indices of the components sorted by the x-axis
     */
    std::vector<size_t> sorted_cc_indices() const;
    /**
     *  Returns the size of the group
     */
    cv::Size group_size() const;

    /**
     *  Returns an image of the group
     */
    cv::Mat get_image() const;
    /**
     *  Returns an image of the group of size `size`
     */
    cv::Mat get_image(const cv::Size &size) const;

    /**
     *  Sums up the probabilities in probs which correspond to the connected components in this group.
     *
     *  @param probs is the global list of probabilities
     *  @returns the average probability that the components of this group are text
     */
    float calculate_probability(const std::vector<double> &probs) const;
    
    /**
     *  Returns true if the other_group is overlapping this group with the given threshold
     */
    bool is_overlapping(const CCGroup &other_group, float thresh=0.3) const;
    bool eq(const CCGroup &other_group) const;
    bool contains(const CCGroup &other_group) const;
    /**
     *  Returns true if the other_group is overlapping this group with the given 
     *  threshold by comparing the bounding boxes for each component
     */
    bool is_overlapping_component_bbx(const CCGroup &other_group, float thresh=0.8) const;
    bool share_elements(const CCGroup &grp) const;

    std::vector<CC> ccs;
};

}


#endif /* end of include guard: CCGROUP_H */
