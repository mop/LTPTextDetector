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
#include <text_detector/MserDetector.h>

#include <boost/timer/timer.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ml/ml.hpp>
#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include <text_detector/BinaryMaskExtractor.h>
//#include <text_detector/CCUtils.h>
#include <text_detector/CNN.h>
#include <text_detector/CNNConnectedComponentClassifier.h>
#include <text_detector/ConfigurationManager.h>
#include <text_detector/ConnectedComponentGrouper.h>
#include <text_detector/CRFLinConnectedComponentFilterer.h>
#include <text_detector/CRFRFConnectedComponentFilterer.h>
#include <text_detector/HardPPWordSplitter.h>
#include <text_detector/LibSVMClassifier.h>
#include <text_detector/ModelManager.h>
#include <text_detector/MserExtractorFast.h>
#include <text_detector/MserExtractor.h>
#include <text_detector/RFConnectedComponentClassifier.h>
#include <text_detector/RFConnectedComponentFilterer.h>
#include <text_detector/SimpleWordSplitter.h>
#include <text_detector/SoftPPWordSplitter.h>
#include <text_detector/SVMConnectedComponentClassifier.h>
#include <text_detector/SVMRFConnectedComponentClassifier.h>

namespace TextDetector {

MserDetector::MserDetector(const std::shared_ptr<ConfigurationManager> &mgr)
: _config_manager(mgr), _model_manager(new ModelManager(mgr))
{
}

MserDetector::~MserDetector()
{
}

static inline void append(cv::Mat &to, const cv::Mat &from)
{
    cv::Mat cpy(to.rows + from.rows, to.cols, CV_32FC1);

    cv::Mat submat = cpy.rowRange(0, to.rows);
    to.copyTo(submat);

    submat = cpy.rowRange(to.rows, to.rows + from.rows);
    from.copyTo(submat);
    to = cpy;
}

template <class T>
static inline void append(T &to, const T &from)
{
    to.insert(to.end(), from.begin(), from.end());
}

std::shared_ptr<ConnectedComponentFilterer>
MserDetector::get_connected_component_filter(
    const cv::Mat &train_image,
    const cv::Mat &grad_image,
    const cv::Mat &unary_features) const
{
    switch (ConfigurationManager::instance()->get_classification_model()) {
        case ConfigurationManager::CLASSIFICATION_MODEL_RANDOM_FOREST:
            return std::shared_ptr<ConnectedComponentFilterer>(
            	new RFConnectedComponentFilterer(0, _config_manager->get_pre_classification_prob_threshold()));
        case ConfigurationManager::CLASSIFICATION_MODEL_CRF_LIN:
            return std::shared_ptr<ConnectedComponentFilterer>(
                new CRFLinConnectedComponentFilterer(train_image,
                		grad_image, unary_features, _model_manager->get_graph_labeler()));
        case ConfigurationManager::CLASSIFICATION_MODEL_CRF_RF:
        default:
            return std::shared_ptr<ConnectedComponentFilterer>(
                new CRFRFConnectedComponentFilterer(
                    train_image,
                    grad_image,
                    _model_manager->get_graph_labeler(),
                    _model_manager->get_pairwise_1_1_classifier(),
                    _model_manager->get_pairwise_1_0_classifier(),
                    _model_manager->get_pairwise_0_0_classifier()));
    }
}

std::shared_ptr<TextDetector::ConnectedComponentClassifier>
MserDetector::get_connected_component_classifier() const
{
	switch (_config_manager->get_preclassification_model()) {
	case ConfigurationManager::PRE_CLASSIFICATION_MODEL_CNN:
		return std::make_shared<TextDetector::CNNConnectedComponentClassifier>(
            _model_manager->get_unary_cnn_classifier());
	case ConfigurationManager::PRE_CLASSIFICATION_MODEL_RANDOM_FOREST:
		return std::make_shared<TextDetector::RFConnectedComponentClassifier>(
            _model_manager->get_unary_random_forest_classifier());
	case ConfigurationManager::PRE_CLASSIFICATION_MODEL_RF_SVM_ENSEMBLE:
	{
		std::shared_ptr<TextDetector::RFConnectedComponentClassifier> rf =
            std::make_shared<TextDetector::RFConnectedComponentClassifier>(
				_model_manager->get_unary_random_forest_classifier());
		std::shared_ptr<TextDetector::SVMConnectedComponentClassifier> svm =
            std::make_shared<TextDetector::SVMConnectedComponentClassifier>(
				_model_manager->get_svm_classifier());
		return std::make_shared<TextDetector::SVMRFConnectedComponentClassifier>(
            rf, svm);
	}
	case ConfigurationManager::PRE_CLASSIFICATION_MODEL_SVM:
		return std::make_shared<TextDetector::SVMConnectedComponentClassifier>(
            _model_manager->get_svm_classifier());
	default:
		assert(0);
		break;
	}
	return nullptr;
}

void MserDetector::extract_components_on_channels(
	const cv::Mat &input_image,
	const cv::Mat &gradient_image,
	const std::vector<cv::Mat> &img_channels,
	const cv::Mat &detector_mask,
    std::vector<double> &all_probs,
    std::vector<std::vector<double> > &all_per_classifier_probs,
    cv::Mat &all_unary_features,
    std::vector<std::pair<int, std::vector<cv::Point> > > &all_comps,
    std::vector<MserElement> &all_elements) const
{
    std::shared_ptr<TextDetector::ConnectedComponentClassifier> clf(
        get_connected_component_classifier());

	boost::timer::cpu_timer t;
    for (int chan = 0; chan < img_channels.size(); chan++) {
        cv::Mat train_image_gray = img_channels[chan];

        std::vector<double> probs;
        std::vector<std::vector<double> > per_classifier_probs;
        std::vector<std::pair<int, std::vector<cv::Point> > > comps;
        cv::Mat unary_features;
        std::vector<MserElement> elements;
        int start_idx = all_probs.size();

        if (!_config_manager->include_binary_masks() ||
            chan < img_channels.size() - 1) {
            TextDetector::MserExtractorFast extractor(
                input_image,
                train_image_gray,
                detector_mask,
                gradient_image,
                clf,
                all_probs.size());

            extractor.extract(
                unary_features,
                probs,
                per_classifier_probs,
                comps, elements);
        } else {
            TextDetector::BinaryMaskExtractor extractor(
                input_image,
                train_image_gray,
                detector_mask,
                gradient_image,
                clf,
                all_probs.size()
            );
            extractor.extract(
                unary_features,
                probs,
                per_classifier_probs,
                comps, elements);
        }

        t.start();
        append(all_unary_features, unary_features);
        append(all_probs, probs);
        append(all_per_classifier_probs, per_classifier_probs);
        append(all_comps, comps);
        append(all_elements, elements);
        if (_config_manager->verbose())
            std::cout << "Accumulated CCs " << all_comps.size() <<  " in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
        //cv::Mat img(input_image.rows, input_image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        //for (int i = 0; i < all_comps.size(); i++) {
        //
        //    int idx = all_comps[i].first;
        //    float proba = all_probs[idx];
        //    int greenish = proba * 255;
        //    if (proba <= 0.0) continue;
        //    for (cv::Point pt : all_comps[i].second) {
        //        img.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, greenish, 255-greenish);
        //    }
        //}
        //cv::imshow("Connected components", img);
        //cv::waitKey(0);

        t.start();
        if (start_idx > 0)
            filter_overlapping_components(all_comps, all_elements, all_probs);
        if (_config_manager->verbose())
            std::cout << "Merged overlapping CCs: " << all_comps.size() << "  in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    }
}

static float do_intersect_fast(const MserElement &el1, const MserElement &el2, bool reverse=false)
{
    float area1 = el1.get_bounding_rect().width * el1.get_bounding_rect().height;
    float area2 = el2.get_bounding_rect().width * el2.get_bounding_rect().height;
    const MserElement &smaller = area1 < area2 ? el1 : el2;
    const MserElement &bigger = area1 >= area2 ? el1 : el2;
    std::vector<cv::Point> px1 = smaller.get_pixels();
    std::vector<cv::Point> px2 = bigger.get_pixels();
    cv::Rect smaller_rect = smaller.get_bounding_rect();
    cv::Rect bigger_rect = bigger.get_bounding_rect();

    cv::Mat bitmap(smaller.get_bounding_rect().height, smaller.get_bounding_rect().width, CV_8UC1, cv::Scalar(0));
    for (cv::Point p : px1) {
        bitmap.at<unsigned char>(p.y - smaller_rect.y, p.x - smaller_rect.x) = 255;
    }

    float area = 0;
    for (cv::Point p : px2) {
        int x = p.x - smaller_rect.x;
        int y = p.y - smaller_rect.y;
        if (x < 0 || y < 0 || x >= smaller_rect.width || y >= smaller_rect.height) continue;

        if (bitmap.at<unsigned char>(y,x)) {
            area += 1.0f;
        }
    }
    if (reverse) {
        return area / std::min(px1.size(), px2.size());
    } else {
        return area / std::max(px1.size(), px2.size());
    }
}

void MserDetector::filter_overlapping_components(
    std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
    const std::vector<MserElement> &all_elements,
    const std::vector<double> &probs) const
{
    // because of parallelism std::vector<_int_> is needed instead of std::vector<_bool_>
    std::vector<int> remove_mask(comps.size(), 0);
    #pragma omp parallel for
    for (int i = 0; i < comps.size(); i++) {
        cv::Rect bbox_i = all_elements[comps[i].first].get_bounding_rect();
        int area_i = bbox_i.width * bbox_i.height;
        for (int j = i+1; j < comps.size(); j++) {
            cv::Rect bbox_j = all_elements[comps[j].first].get_bounding_rect();
            cv::Rect intersect = bbox_i & bbox_j;
            cv::Rect rect_union = bbox_i | bbox_j;
            int area_intersect = intersect.width * intersect.height;
            int area_j = bbox_j.width * bbox_j.height;
            int area_union = rect_union.width * rect_union.height;

            if (area_intersect == 0) continue;

            if (float(area_intersect) / std::max(area_i, area_j) > 0.4) {
                if (do_intersect_fast(all_elements[comps[i].first], all_elements[comps[j].first]) > 0.4) {
                    remove_mask[probs[comps[i].first] > probs[comps[j].first] ? j : i] = true;
                }
            }
        }
    }

    std::vector<std::pair<int, std::vector<cv::Point> > > result;
    result.reserve(comps.size());
    for (int i = 0; i < comps.size(); i++) {
        if (!remove_mask[i])
            result.push_back(comps[i]);
    }

    comps = result;
}


std::shared_ptr<WordSplitter>
MserDetector::get_word_splitter() const
{
    switch (_config_manager->get_word_split_model()) {
        case ConfigurationManager::MODEL_PROJECTION_PROFILE:
            return std::shared_ptr<HardPPWordSplitter>(
                new HardPPWordSplitter(
                    _config_manager->allow_single_letters(),
                    false));
                    //ConfigurationManager::instance()->verbose()));
        case ConfigurationManager::MODEL_SIMPLE:
            return std::shared_ptr<SimplePPWordSplitter>(
                new SimplePPWordSplitter(
                    _config_manager->allow_single_letters(),
                    false));
                    //ConfigurationManager::instance()->verbose()));
        case ConfigurationManager::MODEL_PROJECTION_PROFILE_SOFT:
        default:
            return std::shared_ptr<SoftPPWordSplitter>(
                new SoftPPWordSplitter(
                    _config_manager->allow_single_letters(),
                    false));
                    //ConfigurationManager::instance()->verbose()));
    }
}

static cv::Mat
show_groups_color(
    const std::vector<CCGroup> &groups,
    const cv::Size &window_size,
    const std::vector<double> &probs,
    bool show=false)
{
    cv::Mat img(window_size.height, window_size.width, CV_8UC3, cv::Scalar(255,255,255));
    for (CCGroup g : groups) {
        img = g.draw(window_size, probs, img, false);
    }
    if (show) {
        cv::imshow("GROUPS", img);
        cv::waitKey(0);
    }
    return img;
}

std::vector<cv::Rect> MserDetector::operator()(
    const cv::Mat &input_image,
    cv::Mat &result_image,
    const cv::Mat &detector_mask,
    const cv::Mat &extra_bin_mask) const
{
	if (!detector_mask.empty() && detector_mask.type() != CV_8UC1) 
		throw std::runtime_error("Unexpected detector mask type");
	if (!extra_bin_mask.empty() && extra_bin_mask.type() != CV_8UC1)
		throw std::runtime_error("Unexpected extra_bin_mask type");
	if (input_image.type() != CV_8UC3)
		throw std::runtime_error("Unexpected input image type");
	if (!detector_mask.empty() && (detector_mask.rows != input_image.rows ||
								   detector_mask.cols != input_image.cols) )
		throw std::runtime_error(
			"rows and cols of input image must match detector mask");
	if (!extra_bin_mask.empty() && (extra_bin_mask.rows != input_image.rows ||
									extra_bin_mask.cols != input_image.cols))
		throw std::runtime_error(
			"rows and cols of input image must match extra binary mask");

	cv::Mat mask = detector_mask;
    if (mask.empty()) {
        mask = cv::Mat(
            input_image.rows, input_image.cols, CV_8UC1, cv::Scalar::all(255));
	}

    cv::Mat img_luv;
    cv::cvtColor(input_image, img_luv, CV_RGB2Luv);
    std::vector<cv::Mat> chans;
    cv::split(img_luv, chans);

    cv::Mat image_gray;
    cv::cvtColor(input_image, image_gray, CV_RGB2GRAY);
    cv::Mat gradient_image = compute_gradient(input_image);

    std::vector<cv::Mat> img_channels;
    if (!_config_manager->ignore_gray()) {
        img_channels.push_back(image_gray);
    }
    if (!_config_manager->ignore_color()) {
        img_channels.push_back(chans[1]);
        img_channels.push_back(chans[2]);
    }

    if (!extra_bin_mask.empty())
        img_channels.push_back(extra_bin_mask);


    std::vector<double> all_probs;
    std::vector<std::vector<double> > all_per_classifier_probs;
    cv::Mat all_unary_features;
    std::vector<std::pair<int, std::vector<cv::Point> > > all_comps;
    std::vector<MserElement> all_elements;

    extract_components_on_channels(input_image, gradient_image,
        img_channels, mask,
    	all_probs, all_per_classifier_probs, all_unary_features,
    	all_comps, all_elements);

    boost::timer::cpu_timer t;
    all_comps = get_connected_component_filter(
        input_image, gradient_image, all_unary_features
    )->filter_compontents(all_comps, all_elements, all_per_classifier_probs);

    //cv::Mat img(train_image.rows, train_image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    //for (int i = 0; i < all_comps.size(); i++) {
    //
    //    int idx = all_comps[i].first;
    //    float proba = all_probs[idx];
    //    int greenish = proba * 255;
    //    if (proba <= 0.0) continue;
    //    for (cv::Point pt : all_comps[i].second) {
    //        img.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, greenish, 255-greenish);
    //    }
    //}
    ////cv::imwrite("IMFG2.png", img);
    //cv::imshow("Connected components", img);
    //cv::waitKey(0);

    if (_config_manager->verbose()) {
        std::cout << "Filtered components in "
        		  << boost::timer::format(t.elapsed(), 5, "%w")
        		  << std::endl;
    }

    t.start();
    std::vector<CCGroup> groups;
    ConnectedComponentGrouper grouper;
    grouper(input_image, gradient_image, all_probs,
    		all_elements, all_comps, groups);
    if (_config_manager->verbose()) {
        std::cout << "Grouped components in "
        		  << boost::timer::format(t.elapsed(), 5, "%w")
                  << std::endl;
    }


    // split into words...
    std::vector<cv::Rect> words;
    if (!_config_manager->ignore_word_splitting()) {
        t.start();
        std::shared_ptr<WordSplitter> splitter(get_word_splitter());
        for (int i = 0; i < groups.size(); i++) {
            std::vector<cv::Rect> w(splitter->split(groups[i]));
            words.insert(words.begin(), w.begin(), w.end());
        }
        if (_config_manager->verbose()) {
            std::cout << "Split into words in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
        }
    } else {
        words.reserve(groups.size());
        for (size_t i = 0; i < groups.size(); i++) {
            words.push_back(groups[i].get_rect());
        }
    }

    result_image = show_groups_color(
        groups,
        cv::Size(input_image.cols, input_image.rows),
        all_probs, false);

	return words;
}

} /* namespace TextDetector */
