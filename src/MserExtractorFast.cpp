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
#include <text_detector/MserExtractorFast.h>

#include <unordered_map>
#include <boost/timer/timer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include <text_detector/config.h>
#include <text_detector/ConfigurationManager.h>
#include <text_detector/ConnectedComponentClassifier.h>
#include <text_detector/mser.h>
#include <text_detector/MserTree.h>

namespace TextDetector {
MserExtractorFast::MserExtractorFast(
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

static inline cv::Mat get_binary_image(const cv::Mat &gray_img, const MSER::Region &region, bool inverse = false)
{
    int y = region.pixel_ / gray_img.cols;
    int x = region.pixel_ % gray_img.cols;
    int gray = gray_img.at<unsigned char>(y,x);
    cv::Mat roi = gray_img.rowRange(region.start_y_, region.end_y_+1).colRange(region.start_x_, region.end_x_+1);
    cv::Mat bin = inverse ? roi <= gray : roi >= gray;
    bin = bin - 128;
    cv::floodFill(bin, cv::Point(x-region.start_x_, y-region.start_y_), 255, 0, cv::Scalar(), cv::Scalar(), 4);
    bin = bin == 255;
    return bin;
}

static inline std::vector<cv::Point> get_pixels(const cv::Mat &gray_img, const MSER::Region &reg, bool inverse = false)
{
    cv::Mat bin_img = get_binary_image(gray_img, reg, inverse);
    std::vector<cv::Point> pixel_list;
    pixel_list.reserve(bin_img.rows*bin_img.cols);
    for (int i = 0; i < bin_img.rows; i++) {
        for (int j = 0; j < bin_img.cols; j++) {
            if (bin_img.at<unsigned char>(i,j)) {
                pixel_list.push_back(cv::Point(j+reg.start_x_,i+reg.start_y_));
            }
        }
    }
    return pixel_list;
}

void MserExtractorFast::create_uid_to_index_map(
		std::vector<MSER::Region> regions[2],
		std::unordered_map<int, int> uid_to_index[2]) {
	for (int j = 0; j < 2; j++) {
		int offset = j == 0 ? 0 : regions[0].size();
		for (size_t i = 0; i < regions[j].size(); i++) {
			assert(regions[j][i].uid_ >= 0);
			uid_to_index[j].insert(
					std::make_pair(regions[j][i].uid_, i + offset));
		}
	}
}

void MserExtractorFast::compute_features(const cv::Mat& swt1,
		const cv::Mat& swt2, MserElement& el) {
	el.compute_features(_image_color, _gradient_image, swt1, swt2);
	if (ConfigurationManager::instance()->get_preclassification_model()
			== ConfigurationManager::PRE_CLASSIFICATION_MODEL_CNN)
		el.compute_hog_features(_image_gray);
}

void MserExtractorFast::compute_probs(
	size_t idx, const MserElement &el,
	std::vector<double> &probs,
	std::vector<std::vector<double> > &per_classifier_probs)
{
    cv::Mat f = el.get_unary_features();
    double prob;
    std::vector<double> v;

    cv::Mat bin_image = el.get_binary_image();
    if (!bin_image.empty()) {
        bin_image = bin_image.reshape(0, 28);
    }
    _classifier->classify(f, bin_image, prob, v);
    probs[idx] = prob;
    per_classifier_probs[idx] = v;
}

void MserExtractorFast::extract(
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

    // this mser detector is significantly faster than the OpenCV one on
    // large images!
	MSER mser(false, 3, 10.0 / (_image_gray.rows * _image_gray.cols), 1.0, 0.50, 0.20);
    std::vector<MSER::Region> regions[2];
    cv::Mat inv_img = 255 - _image_gray;
    mser(_image_gray.ptr<uint8_t>(0,0),
    	_image_gray.cols, _image_gray.rows, regions[0]);
    mser(inv_img.ptr<uint8_t>(0,0),
    	_image_gray.cols, _image_gray.rows, regions[1]);

    int region_size = regions[0].size() + regions[1].size();
    
    if (ConfigurationManager::instance()->verbose())
        std::cout << "Extracted " << region_size << " MSERs in "
        		  << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

    if (ConfigurationManager::instance()->keep_unary_features())
        unary_features = cv::Mat(region_size,
            N_UNARY_FEATURES, CV_32FC1, cv::Scalar(0.0));

    all_elements.resize(region_size);
    per_classifier_probs.resize(region_size);
    probs.resize(region_size);

    t.start();

    std::unordered_map<int,int> uid_to_index[2];
	create_uid_to_index_map(regions, uid_to_index);
    std::cout << uid_to_index[0].size() << std::endl;

    std::vector<std::vector<cv::Point> > pixels(region_size);
    std::vector<cv::Vec4i> hierarchy(region_size);
    // black on white and white on black
    for (int j = 0; j < 2; j++) {
        int offset = j == 0 ? 0 : regions[0].size();
        #pragma omp parallel for
        for (size_t i = 0; i < regions[j].size(); i++) {
            pixels[i+offset]    = get_pixels(_image_gray, regions[j][i], j == 0);
            // the hierarchy has the index as first element and the parent
            // as last
            hierarchy[i+offset] = cv::Vec4i(
            	uid_to_index[j][regions[j][i].uid_], -1, -1,
            	regions[j][i].parent_uid_ == -1 ?
            		-1 : uid_to_index[j][regions[j][i].parent_uid_]);
            MserElement el(-1, -1, -1, pixels[i+offset]);

            // really be bigger than 2x5 pixels -> otherwise it is no CC
            if (is_component_invalid(_mask, el.get_bounding_rect())) {
                probs[i+offset] = 0.0;
                per_classifier_probs[i+offset] = std::vector<double> (2,0.0);
            } else {
				compute_features(swt1, swt2, el);
				compute_probs(i + offset, el, probs, per_classifier_probs);
            }
            all_elements[i+offset] = el;
        }
    }

    if (ConfigurationManager::instance()->verbose())
        std::cout << "Classified CCs in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;;

    // prune hierarchical by using the probabilities of the random forest

    t.start();
    MserTree tree(pixels, probs, hierarchy);//, mask, 0.01*255.0f);
    tree.linearize();
    tree.accumulate();

    std::vector<std::vector<cv::Point> > components =
        tree.get_accumulated_contours();
    std::vector<int> idxs = tree.get_accumulated_indices();

    if (ConfigurationManager::instance()->verbose()) {
        std::cout << "Eliminated duplicates in component tree to "
        		  << components.size() << " in "
        		  << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    }

    comps.reserve(components.size());

    assert(idxs.size() == components.size());

    for (size_t i = 0; i < components.size(); i++) {
        if (probs[idxs[i]] <= 0.0f) continue;
        comps.push_back(std::make_pair(idxs[i] + _uid_offset, components[i]));
    }
}
}

