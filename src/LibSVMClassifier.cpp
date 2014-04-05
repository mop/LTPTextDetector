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
#include <text_detector/LibSVMClassifier.h>

#include <opencv2/ml/ml.hpp>

namespace TextDetector {

LibSVMClassifier::LibSVMClassifier(const std::string &filename)
: _svm_model(0)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    std::string sigmoid_scale_file, means_file, stds_file, model_file;

    fs["sigmoid_scale_file"] >> sigmoid_scale_file;
    fs["libsvm_model_file"] >> model_file;
    fs["means_file"] >> means_file;
    fs["stds_file"] >> stds_file;
    fs["log_dimensions"] >> _log_dimensions;

    load_means(means_file);
    load_stds(stds_file);
    load_sigmoid(sigmoid_scale_file);
    load_libsvm_model(model_file);
}

LibSVMClassifier::~LibSVMClassifier() 
{
    //if (_svm_model)
    //    svm_free_and_destroy_model(&_svm_model);
}

static cv::Mat load(const std::string &filename)
{
    cv::TrainData data;
    data.read_csv(filename.c_str());
    data.set_delimiter(',');
    return cv::Mat(data.get_values()).clone();
}

void LibSVMClassifier::load_means(const std::string &means_file)
{
    _means = load(means_file);
}

void LibSVMClassifier::load_stds(const std::string &stds_file)
{
    _stds = load(stds_file);
} 

void LibSVMClassifier::load_sigmoid(const std::string &sigmoid_scale_file)
{
    cv::Mat tmp = load(sigmoid_scale_file);
    _sigmoid_scale = tmp.at<float>(0,0);
    if (tmp.rows > 1) {
        _sigmoid_intercept = tmp.at<float>(1,0);
    } else {
        _sigmoid_intercept = 0.0f;
    }
}

static void convert_to_libsvm(const cv::Mat &row, struct svm_node *nodes)
{
    for (int i = 0; i < row.cols; i++) {
        nodes[i].value = row.at<float>(0,i);
        nodes[i].index = i+1;
    }
    nodes[row.cols].value = 0;
    nodes[row.cols].index = -1;
}

cv::Mat LibSVMClassifier::preprocess(const cv::Mat &vecs) const
{
    cv::Mat result(vecs.rows, vecs.cols, CV_32FC1);
    for (int i = 0; i < vecs.rows; i++) {
        cv::Mat src_row = vecs.row(i);
        cv::Mat dst_row = result.row(i);
        src_row.copyTo(dst_row);
        for (int j = 0; j < _log_dimensions.size(); j++) {
            int dim = _log_dimensions[j];
            dst_row.at<float>(0,dim) = cv::log(std::max(dst_row.at<float>(0,dim), 1e-5f));
        }

        dst_row = dst_row - _means;
        cv::divide(dst_row, _stds, dst_row);
    }
    return result;
}

cv::Mat LibSVMClassifier::predict_probs(const cv::Mat &vecs) const
{
    cv::Mat processed = preprocess(vecs);
    cv::Mat result(vecs.rows, 1, CV_32FC1);

    for (int i = 0; i < result.rows; i++) {
        double dec_values[2];
        struct svm_node nodes[processed.cols+1];
        convert_to_libsvm(processed.row(i), nodes);
        svm_predict_values(_svm_model, nodes, dec_values);
        result.at<float>(i, 0) = 1.0/(1.0+exp(-(_sigmoid_scale*dec_values[0] + _sigmoid_intercept)));
    }
    return result;
}

void LibSVMClassifier::load_libsvm_model(const std::string &filename)
{
    _svm_model = svm_load_model(filename.c_str());
}

}

