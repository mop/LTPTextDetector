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
#include <text_detector/MserExtractor.h>

#include <boost/timer/timer.hpp>
#include <opencv2/core/core.hpp>
#include <cassert>
#include <iostream>
#include <string>

#include <text_detector/config.h>
#include <text_detector/CacheManager.h>
#include <text_detector/ConfigurationManager.h>
#include <text_detector/ConnectedComponentClassifier.h>
#include <text_detector/HierarchicalMSER.h>
#include <text_detector/MserTree.h>

namespace TextDetector {

MserExtractor::MserExtractor(
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

MserExtractor::~MserExtractor()
{
}

void MserExtractor::extract_msers(const boost::timer::cpu_timer& t,
		std::vector<std::vector<cv::Point> > &msers,
		std::vector<double> &probs,
		std::vector<cv::Vec4i> &hierarchy)
{
	cv::HierarchicalMSER mser(1, 1, 14400000, 0.5, 0.1, true);
	// load the msers from cache if available - otherwise extract them
	bool msers_loaded = true;
	if (!ConfigurationManager::instance()->has_cache()) {
		mser(_image_gray, msers, probs, hierarchy);
		msers_loaded = false;
	} else if (!CacheManager::instance()->get_msers(msers, hierarchy,
			_uid_offset)) {
		mser(_image_gray, msers, probs, hierarchy);
		msers_loaded = false;
	}

	if (ConfigurationManager::instance()->verbose())
		std::cout << "Extracted " << msers.size() << " MSERs in "
				<< boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

	if (ConfigurationManager::instance()->has_cache() && !msers_loaded) {
		CacheManager::instance()->set_msers(msers, hierarchy, _uid_offset);
	}
}

void MserExtractor::compute_features(const cv::Mat& swt1, const cv::Mat& swt2,
		size_t i, MserElement& el) {

	if (!ConfigurationManager::instance()->has_cache()) {
		el.compute_features(_image_color, _gradient_image, swt1, swt2);
	} else {
        #pragma omp critical
		{
			// If we have a cache we serialize/unserialize things here.
			// This caching mechanism is somewhat broken by design and a big
			// and dirty hack!
			// We only need to store (and deserialize) the mean-stroke-width
			// here because classification-results based on unary features are
			// also cached. We need the stroke-width, however, for the pairwise
			// feature calculation! Other features such as area/perimeter^2
			// are not needed for pairwise classification and hence not
			// serialized! This also implies that el.get_unary_feature()
			// returns just garbage! We don't serialize here everything since
			// that is just too slow.
			if (CacheManager::instance()->has_feature_entry(i + _uid_offset)) {
				float stroke_width = CacheManager::instance()->query_feature(
						i + _uid_offset).at(0);
				el.set_raw_swt_mean(stroke_width);
			} else {
				el.compute_features(_image_color, _gradient_image, swt1, swt2);
				std::vector<float> f_vec(1);
				f_vec.at(0) = el.get_raw_swt_mean();
				CacheManager::instance()->set_feature(i + _uid_offset, f_vec);
			}
		}
	}
	if (ConfigurationManager::instance()->get_preclassification_model() ==
        ConfigurationManager::PRE_CLASSIFICATION_MODEL_CNN)
        el.compute_hog_features(_image_gray);

}

void MserExtractor::compute_probs(size_t i, const MserElement &el,
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
    if (!ConfigurationManager::instance()->has_cache()) {
        _classifier->classify(f, bin_image, prob, v);
    } else {
        #pragma omp critical
        {
        if (CacheManager::instance()->has_entry(i + _uid_offset)) {
            v = CacheManager::instance()->query(i + _uid_offset);
            prob = 0;
            for (int j = 0; j < v.size(); j++)
                prob += v[j];
            prob /= v.size();
        } else {
            _classifier->classify(f, bin_image, prob, v);
            CacheManager::instance()->set(i + _uid_offset, v);
        }
        }
    }
    probs[i] = prob;
    per_classifier_probs[i] = v;
}

void MserExtractor::extract(
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
        std::cout << "Computed SWT in: "
                  << boost::timer::format(t.elapsed(), 5, "%w")
                  << std::endl;
    
    t.start(); 
    std::vector<std::vector<cv::Point> > msers;
    std::vector<cv::Vec4i> hierarchy;
	extract_msers(t, msers, probs, hierarchy);

    if (ConfigurationManager::instance()->keep_unary_features())
        unary_features = cv::Mat(msers.size(), N_UNARY_FEATURES, CV_32FC1, cv::Scalar(0.0));

    all_elements.resize(msers.size());
    per_classifier_probs.resize(msers.size());
    probs.resize(msers.size());

    // some sanity checks
    assert(probs.size() == msers.size());
    assert(per_classifier_probs.size() == msers.size());

    t.start();
    #pragma omp parallel for
    for (size_t i = 0; i < msers.size(); i++) {
        MserElement el(-1, -1, -1, msers[i]);

        // really be bigger than 2x5 pixels -> otherwise it is no CC
        if (is_component_invalid(_mask, el.get_bounding_rect())) {
            probs[i] = 0.0;
            per_classifier_probs[i] = std::vector<double> (2,0.0);
        } else {
			compute_features(swt1, swt2, i, el);
			compute_probs(i, el, probs, per_classifier_probs);
        }
        all_elements[i] = el;
    }

    if (ConfigurationManager::instance()->verbose())
        std::cout << "Classified CCs in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;;

    // prune hierarchical by using the probabilities of the random forest
    t.start();
    MserTree tree(msers, probs, hierarchy);//, mask, 0.01*255.0f);
    tree.linearize();
    tree.accumulate();

    std::vector<std::vector<cv::Point> > components = tree.get_accumulated_contours();
    std::vector<int> idxs = tree.get_accumulated_indices();

    if (ConfigurationManager::instance()->verbose()) {
        std::cout << "Eliminated duplicates in component tree to " << components.size() << " in " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;;
    }

    comps.reserve(components.size());
    assert(idxs.size() == components.size());

    for (size_t i = 0; i < components.size(); i++) {
        comps.push_back(std::make_pair(idxs[i] + _uid_offset, components[i]));
    }
}

}
