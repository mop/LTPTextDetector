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

#ifndef LIBSVMCLASSIFIER_H
#define LIBSVMCLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <svm.h>

namespace TextDetector {

/**
 * This class is a wrapper around libsvm. It normalizes input vectors vor libsvm, 
 * does some other preprocessing and returns callibrated probabilities.
 */
class LibSVMClassifier 
{
public:
    /**
     *  Constructs the classifier from the given .yml model file
     */
    LibSVMClassifier(const std::string &model_file);
    ~LibSVMClassifier();

    /**
     *  Predict the probabilities from the given row vectors
     */
    cv::Mat predict_probs(const cv::Mat &vectors) const;
    /**
     *  Predicts the probability of the given row-vector.
     */
    float predict_prob(const cv::Mat &f) const {
        cv::Mat p = predict_probs(f);
        return p.at<float>(0,0);
    }

private:
    /**
     *  Preprocesses the given vector
     */
    cv::Mat preprocess(const cv::Mat &vectors) const;

    /**
     *  Loads the libsvm model.
     */
    void load_libsvm_model(const std::string &model);
    /**
     *  Loads the means used for mean-normalization
     */
    void load_means(const std::string &means);
    /**
     *  Loads the standard deviations used for normalization
     */
    void load_stds(const std::string &means);
    /**
     *  Loads the sigmoid parameters used for normalization
     */
    void load_sigmoid(const std::string &sigmoid);

    //! The means
    cv::Mat _means;
    //! The standard deviations
    cv::Mat _stds;
    //! The scale factor for the sigmoid
    float _sigmoid_scale;
    //! The intercept factor for the sigmoid function
    float _sigmoid_intercept;
    //! The dimensions which should be logarithmized
    std::vector<int> _log_dimensions;

    //! Pointer to the svm model of libsvm
    struct svm_model *_svm_model;
};

}

#endif /* end of include guard: LIBSVMCLASSIFIER_H */
