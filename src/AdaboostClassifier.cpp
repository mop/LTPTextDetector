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
#include <text_detector/AdaboostClassifier.h>
#include <text_detector/LTPComputer.h>
#include <text_detector/utils.h>
#include <text_detector/config.h>

#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace TextDetector {
AdaboostClassifier::AdaboostClassifier(const std::string &model_path, float scale_ratio, int num_scales,
                                       int window_width, int window_height, int shift_width, int shift_height)
: _clf(new Detector::Adaboost()),
   _scale_ratio(scale_ratio), _num_scales(num_scales), _window_width(window_width), _window_height(window_height),
  _shift_width(shift_width), _shift_height(shift_height)
{
    std::ifstream ifs(model_path);
    boost::archive::text_iarchive ia(ifs);
    ia >> *_clf;
}

void AdaboostClassifier::detect(const cv::Mat &image, cv::Mat &response)
{
    std::list<cv::Mat> fps;
    bootstrap(image, response, fps, 0.1f, false);
}

void AdaboostClassifier::detect_single_scale(const cv::Mat &image, cv::Mat &response)
{
    std::list<cv::Mat> fps;
    bootstrap_single_scale(image, response, fps, 0, 0.1f, false);
}

void AdaboostClassifier::bootstrap(const cv::Mat &image, cv::Mat &response, std::list<cv::Mat> &false_positives, float thresh, bool sample_false_positives)
{
    // classify each scale...
    cv::Size original_size(image.cols, image.rows);
    std::vector<cv::Mat> results;
    cv::Mat img = image;
    for (int i = 0; i < _num_scales; ++i) {

        std::cout << "scale: " << i << std::endl;
        if (img.rows <= _window_height || img.cols <= _window_width) break;

        cv::Mat result;
        bootstrap_single_scale(
            img,
            result,
            false_positives, 
            i,
            thresh, 
            sample_false_positives);
        results.push_back(result);

        cv::resize(img, img, cv::Size(), _scale_ratio, _scale_ratio);
    }

    response = cv::Mat(original_size.height, original_size.width, CV_32FC1, cv::Scalar(0.0f));
    for (unsigned int i = 0; i < results.size(); ++i) {
        cv::Mat tmp;
        cv::GaussianBlur(results[i], results[i], cv::Size(5,5), 2);
        cv::resize(results[i], tmp, cv::Size(original_size.width, original_size.height), 0, 0, cv::INTER_NEAREST);
        response = response + tmp;
    }
    double min, max;
    cv::minMaxIdx(response, &min, &max);
    cv::multiply(response, cv::Scalar::all(255.0/max), response);
    response.convertTo(response, CV_8UC1);
}

void AdaboostClassifier::bootstrap_single_scale(
        const cv::Mat &image,
        cv::Mat &response,
        std::list<cv::Mat> &false_positives,
        int scale,
        float thresh, bool sample_false_positives)
{
    double img_width = image.cols;
    double img_height = image.rows;

    int n_width_shifts = ceil((img_width - _window_width)    / static_cast<float>(_shift_width)) + 1;
    int n_height_shifts = ceil((img_height - _window_height) / static_cast<float>(_shift_height)) + 1;

    TextDetector::LTPComputer ltp;
    cv::Mat ltp_maps = ltp.compute(image);
    ltp_maps.convertTo(ltp_maps, CV_64FC(16));
    std::vector<cv::Mat> feature_maps;
    cv::split(ltp_maps, feature_maps);

    cv::Mat result_image(image.rows, image.cols, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat norm_image(image.rows, image.cols, CV_32FC1, cv::Scalar(0.0));

    // go through all shifts
    #pragma omp parallel for
    for (int i = 0; i < n_height_shifts; ++i) {
        int y = i*_shift_height;

        for (int j = 0; j < n_width_shifts; ++j) {
            int x = j * _shift_width;
            int actual_x = x;
            int actual_y = y;
            if ((x + _window_width) >= img_width) {
                actual_x = img_width - _window_width;
            }
            if ((y + _window_height) >= img_height) {
                actual_y = img_height - _window_height;
            }

            float result = 1.0/(1.0+exp(-_clf->predict(feature_maps, actual_x, actual_y, Detector::Adaboost::LAZY) + 0)) * 2.0 - 1.0;

            #pragma omp critical
            {
            norm_image.rowRange(actual_y, actual_y + _window_height).colRange(actual_x, actual_x + _window_width) += cv::Scalar(1.0);
            if (result > 0) {
                result_image.rowRange(actual_y, actual_y + _window_height).colRange(actual_x, actual_x + _window_width) += cv::Scalar(result);
                
                if (sample_false_positives && result > thresh) {
                    cv::Mat f = ltp.get_vector<double>(ltp_maps, actual_x, actual_y, actual_x + _window_width, actual_y + _window_height);
                    cv::Mat fps_vec(1, f.cols + 5, CV_32FC1, cv::Scalar(0.0f));
                    cv::Mat sub = fps_vec.colRange(5,fps_vec.cols);
                    f.copyTo(sub);

                    fps_vec.at<float>(0, 1) = result;
                    fps_vec.at<float>(0, 2) = scale;
                    fps_vec.at<float>(0, 3) = actual_x;
                    fps_vec.at<float>(0, 4) = actual_y;
                    false_positives.push_back(fps_vec);
                }
            }
            }
        }
    }

    norm_image += cv::Scalar(1e-10);
    for (int i = 0; i < result_image.rows; i++) {
        for (int j = 0; j < result_image.cols; j++) {
            result_image.at<float>(i,j) /= norm_image.at<float>(i,j);
        }
    }

    //result_image.copyTo(response);
    response = result_image;
}


}
