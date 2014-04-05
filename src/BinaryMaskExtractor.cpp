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
#include <text_detector/BinaryMaskExtractor.h>

#include <boost/timer/timer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stddef.h>
#include <iostream>
#include <string>

#include <text_detector/config.h>
#include <text_detector/ConfigurationManager.h>
#include <text_detector/ConnectedComponentClassifier.h>

namespace TextDetector {

BinaryMaskExtractor::BinaryMaskExtractor(
        const cv::Mat &image_color,
        const cv::Mat &image_gray,
        const cv::Mat &mask,
        const cv::Mat &gradient_image,
        const std::shared_ptr<ConnectedComponentClassifier> &clf,
        int uid_offset
): _image_color(image_color),
   _image_gray(image_gray),
   _mask(mask),
   _gradient_image(gradient_image),
   _classifier(clf),
   _uid_offset(uid_offset) {}

void BinaryMaskExtractor::compute_features(const cv::Mat& swt1,
		const cv::Mat& swt2, MserElement& el) const {
	el.compute_features(_image_color, _gradient_image, swt1, swt2);
	if (ConfigurationManager::instance()->get_preclassification_model()
			== ConfigurationManager::PRE_CLASSIFICATION_MODEL_CNN)
		el.compute_hog_features(_image_gray);
}

void BinaryMaskExtractor::compute_probs(const MserElement &el,
		std::vector<double> &probs,
		std::vector<std::vector<double> > &per_classifier_probs) const
{
    cv::Mat f = el.get_unary_features();
    double prob;
    std::vector<double> v;

    cv::Mat bin_image = el.get_binary_image();
    if (!bin_image.empty()) {
        bin_image = bin_image.reshape(0, 28);
    }
    _classifier->classify(f, bin_image, prob, v);
    probs.push_back(prob);
    per_classifier_probs.push_back(v);
}

std::vector<cv::Point> BinaryMaskExtractor::get_pixel_list(cv::Mat& img) {
	std::vector<cv::Point> r;
	for (int k = 0; k < img.rows; k++) {
		for (int l = 0; l < img.cols; l++) {
			if (img.at<unsigned char>(k, l)) {
				r.push_back(cv::Point(l, k));
			}
		}
	}
	return r;
}

void BinaryMaskExtractor::extract(
    cv::Mat &unary_features, 
    std::vector<double> &probs,
    std::vector<std::vector<double> > &per_classifier_probs,
    std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
    std::vector<MserElement> &all_elements
)
{
    boost::timer::cpu_timer t; 
    t.start(); 
    
    probs.clear();
    per_classifier_probs.clear();
    comps.clear();
    all_elements.clear();

    cv::Mat swt1, swt2;
    compute_swt(_image_gray, swt1, swt2);
    if (ConfigurationManager::instance()->verbose())
        std::cout << "Computed SWT in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    
    t.start(); 

    std::vector<std::vector<cv::Point> > msers;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat contour_img;
    _image_gray.copyTo(contour_img);
    contour_img = 255 - contour_img;
    cv::findContours(contour_img, msers, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    
    if (ConfigurationManager::instance()->verbose())
        std::cout << "Extracted "
        		  << msers.size()
        		  << " Contours in "
        		  << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

    if (ConfigurationManager::instance()->keep_unary_features())
        unary_features = cv::Mat(msers.size(), N_UNARY_FEATURES, CV_32FC1, cv::Scalar(0.0));

    all_elements.clear();

    t.start();
    std::vector<std::vector<cv::Point> > regions;
    for (int i = 0; i >= 0;  i = hierarchy[i][0]) {

        cv::Mat img(_image_gray.rows, _image_gray.cols, CV_8UC1, cv::Scalar(0));
        cv::drawContours(img, msers, i, cv::Scalar(255), CV_FILLED, 8, hierarchy, 1);

		std::vector<cv::Point> r(get_pixel_list(img));

        MserElement el(-1, -1, -1, r);
        regions.push_back(r);

        // really be bigger than 2x5 pixels -> otherwise it is no CC
        if (is_component_invalid(_mask, el.get_bounding_rect())) {
            probs.push_back(0.0);
            per_classifier_probs.push_back(std::vector<double> (2,0.0));
        } else {
			compute_features(swt1, swt2, el);
			compute_probs(el, probs, per_classifier_probs);
        }
        all_elements.push_back(el);
    }

    if (ConfigurationManager::instance()->verbose())
        std::cout << "Classified CCs in "
        		  << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

    std::vector<std::vector<cv::Point> > components = regions;

    comps.clear();
    comps.reserve(components.size());

    for (size_t i = 0; i < components.size(); i++) {
        comps.push_back(std::make_pair(i + _uid_offset, components[i]));
    }
}
}

