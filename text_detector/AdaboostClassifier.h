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
#ifndef ADABOOSTCLASSIFIER_H
#define ADABOOSTCLASSIFIER_H

#include <detector/Adaboost.h>
#include <list>

#include <opencv2/core/core.hpp>


namespace TextDetector {
class AdaboostClassifier 
{
public:
    AdaboostClassifier(const std::string &model_path, 
        float scale_ratio = 2/3.0f, int num_scales = 10,
        int window_width = 24, int window_height = 12, 
        int shift_width = 4, int shift_height = 4);

    void detect(const cv::Mat &image, cv::Mat &response);
    void detect_single_scale(const cv::Mat &image, cv::Mat &response);
    void bootstrap(const cv::Mat &image, cv::Mat &response, std::list<cv::Mat> &false_positives, float thresh=0.1f, bool sample_fps=true);
    void bootstrap_single_scale(
        const cv::Mat &image,
        cv::Mat &response,
        std::list<cv::Mat> &false_positives,
        int scale,
        float thresh=0.1f, bool sample_fps=true);

private:
    std::shared_ptr<Detector::Adaboost> _clf;

    float _scale_ratio;
    int _num_scales;
    int _window_width;
    int _window_height;
    int _shift_width;
    int _shift_height;
};
}


#endif /* end of include guard: ADABOOSTCLASSIFIER_H */
