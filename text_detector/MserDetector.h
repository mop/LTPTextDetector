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

#ifndef MSERDETECTOR_H_
#define MSERDETECTOR_H_

#include <opencv2/core/core.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "CCGroup.h"
#include "CCUtils.h"

namespace TextDetector {
class ConnectedComponentClassifier;
class ConnectedComponentFilterer;
class ModelManager;
class WordSplitter;
} /* namespace TextDetector */

namespace TextDetector {

/**
 * This class is responsible for detection text with maximally stable
 * extremal regions.
 */
class MserDetector {
public:
	MserDetector(const std::shared_ptr<ConfigurationManager> &cfg);
	~MserDetector();

	/**
	 * Detects text from a given detector mask and a given input image.
	 *
	 * @param input_image is a CV_8UC3 image
	 * @param detector_mask is a CV_8UC1 image
	 * @param extra_bin_mask is a CV_8UC1 image, which is represents a
	 * 						 extra binary mask used for segmentation
	 * @throws std::runtime_error
	 */
	std::vector<cv::Rect> operator()(
        const cv::Mat &input_image,
        cv::Mat &result_image,
        const cv::Mat &detector_mask  = cv::Mat(),
        const cv::Mat &extra_bin_mask = cv::Mat()) const;
private:
    std::shared_ptr<TextDetector::ConnectedComponentClassifier>
    get_connected_component_classifier() const;
    void extract_components_on_channels(
        const cv::Mat &input_image,
        const cv::Mat &gradient_image,
        const std::vector<cv::Mat> &img_channels,
        const cv::Mat &detector_mask,
        std::vector<double> &all_probs,
        std::vector<std::vector<double> > &all_per_classifier_probs,
        cv::Mat &all_unary_features,
        std::vector<std::pair<int, std::vector<cv::Point> > > &all_comps,
        std::vector<MserElement> &all_elements) const;

    std::shared_ptr<ConnectedComponentFilterer>
    get_connected_component_filter(
        const cv::Mat &train_image,
        const cv::Mat &grad_image,
        const cv::Mat &unary_features) const;
    void group_components(
        const cv::Mat &train_image,
        const cv::Mat &gradient_image,
        const std::vector<double> &probs,
        const std::vector<MserElement> &all_elements,
        std::vector<CCGroup> &groups,
        const std::vector<std::pair<int, std::vector<cv::Point> > > &comps) const;
    void filter_overlapping_components(
        std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        const std::vector<MserElement> &all_elements,
        const std::vector<double> &probs) const;
    std::shared_ptr<WordSplitter> get_word_splitter() const;

	std::shared_ptr<ConfigurationManager> _config_manager;
	std::shared_ptr<ModelManager> _model_manager;
};

} /* namespace TextDetector */

#endif /* MSERDETECTOR_H_ */
