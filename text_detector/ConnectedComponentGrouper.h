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
#ifndef CONNECTEDCOMPONENTGROUPER_H_
#define CONNECTEDCOMPONENTGROUPER_H_

#include <opencv2/core/core.hpp>
#include <utility>
#include <vector>

#include "CCGroup.h"
#include "CCUtils.h"


namespace TextDetector {

class ConnectedComponentGrouper {
public:
	ConnectedComponentGrouper(
		float overlap_threshold=0.6f,
		float distance_threshold=0.0f)
    : _overlap_threshold(overlap_threshold),
      _distance_threshold(distance_threshold) {}
	~ConnectedComponentGrouper() = default;

	/**
	 * Groups the connected components into textlines.
	 *
	 * @param input_image is the RGB input image
	 * @param gradient_image is image-gradient
	 * @param probs are a list of unary probabilities
	 * @param all_elements is a list of mser probabilities
	 * @param comps is a list of tuples [(idx, pixels)] where idx is the
	 *        index into probs and all_elements and pixels is the list
	 *        of image pixels
	 * @param groups [OUT] is the list of connected components
	 */
	void operator()(
        const cv::Mat &input_image,
        const cv::Mat &gradient_image,
        const std::vector<double> &probs,
        const std::vector<MserElement> &all_elements,
        const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        std::vector<CCGroup> &groups) const;
private:

private:
	//! overlap threshold
	float _overlap_threshold;
	//! distance threshold
	float _distance_threshold;

	void create_initial_groups(
			const std::vector<std::pair<int, std::vector<cv::Point> > >& comps,
			const std::vector<MserElement>& all_elements, std::vector<CC>& ccs,
			std::vector<CCGroup>& groups,
			std::vector<MserElement>& elements) const;
	void fill_distance_matrix(const std::vector<CC>& ccs,
			const std::vector<MserElement>& elements,
			const cv::Mat& train_image, const cv::Mat& gradient_image,
			cv::Mat& distance_matrix) const;
	void merge_components(const cv::Mat& distance_matrix,
			std::vector<CCGroup>& groups) const;
	void prune_low_probability_groups(
			const std::vector<double> &probs,
			std::vector<CCGroup> &groups) const;
	void prune_overlapping_groups(
        const std::vector<double> &probs,
		std::vector<CCGroup> &groups) const;
};

} /* namespace TextDetector */

#endif /* CONNECTEDCOMPONENTGROUPER_H_ */
