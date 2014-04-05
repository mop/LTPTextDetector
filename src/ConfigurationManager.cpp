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
#include <text_detector/ConfigurationManager.h>

#include <opencv2/ml/ml.hpp>

namespace TextDetector {
std::shared_ptr<ConfigurationManager> ConfigurationManager::_instance;

ConfigurationManager::ConfigurationManager(const std::string &filename)
: _config_filename(filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    fs["input_directory"] >> _input_directory;
    fs["responses_directory"] >> _responses_directory;
    fs["crf_model_file"] >> _crf_model_file;
    fs["svm_model_file"] >> _svm_model_file;
    fs["cnn_model_file"] >> _cnn_model_file;
    fs["random_forest_model_file"] >> _random_forest_model_file;
    fs["random_forest_pw_1_1_model_file"] >> _random_forest_pw_1_1_model_file;
    fs["random_forest_pw_1_0_model_file"] >> _random_forest_pw_1_0_model_file;
    fs["random_forest_pw_0_0_model_file"] >> _random_forest_pw_0_0_model_file;
    fs["cache_directory"] >> _cache_dir;
    fs["ignore_responses"] >> _ignore_responses;

    fs["random_seed"] >> _random_seed;
    fs["threshold"] >> _threshold;
    fs["word_group_threshold"] >> _word_group_threshold;
    fs["pre_classification_prob_threshold"] >> _pre_classification_prob_threshold;
    fs["allow_single_letters"] >> _allow_single_letters;
    fs["minimum_vertical_overlap"] >> _minimum_vertical_overlap;
    fs["maximum_height_ratio"] >> _maximum_height_ratio;
    fs["ignore_color"] >> _ignore_color;
    fs["ignore_gray"] >> _ignore_gray;
    fs["ignore_word_splitting"] >> _ignore_word_splitting;
    fs["set_gt_prop_to_one"] >> _set_gt_prop_to_one;
    fs["include_binary_masks"] >> _include_binary_masks;
    fs["ignore_grouping_svm"] >> _ignore_grouping_svm;
    fs["min_group_size"] >> _min_group_size;
    if (_min_group_size <= 0) 
        _min_group_size = 3;

    std::string word_split_model, classification_model, pre_classification_model;
    fs["word_split_model"] >> word_split_model;
    fs["classification_model"] >> classification_model;
    fs["pre_classification_model"] >> pre_classification_model;

    _word_split_model = get_word_split_model(word_split_model);
    _classification_model = get_classification_model(classification_model);
    _pre_classification_model = get_pre_classification_model(pre_classification_model);

    fs["verbose"] >> _verbose;

}

int ConfigurationManager::get_word_split_model(const std::string &key) const
{
    if (key == "MODEL_PROJECTION_PROFILE")
        return MODEL_PROJECTION_PROFILE;
    if (key == "MODEL_PROJECTION_PROFILE_SOFT")
        return MODEL_PROJECTION_PROFILE_SOFT;
    if (key == "MODEL_SIMPLE")
        return MODEL_SIMPLE;
    return MODEL_PROJECTION_PROFILE; // default
}

int ConfigurationManager::get_classification_model(const std::string &key) const
{
    if (key == "CLASSIFICATION_MODEL_RANDOM_FOREST")
        return CLASSIFICATION_MODEL_RANDOM_FOREST;
    if (key == "CLASSIFICATION_MODEL_CRF_LIN")
        return CLASSIFICATION_MODEL_CRF_LIN;
    return CLASSIFICATION_MODEL_CRF_RF; // default
}

int ConfigurationManager::get_pre_classification_model(const std::string &key) const
{
    if (key == "PRE_CLASSIFICATION_MODEL_RANDOM_FOREST")
        return PRE_CLASSIFICATION_MODEL_RANDOM_FOREST;
    if (key == "PRE_CLASSIFICATION_MODEL_CNN")
        return PRE_CLASSIFICATION_MODEL_CNN;
    if (key == "PRE_CLASSIFICATION_MODEL_SVM")
        return PRE_CLASSIFICATION_MODEL_SVM;
    if (key == "PRE_CLASSIFICATION_MODEL_RF_SVM_ENSEMBLE")
        return PRE_CLASSIFICATION_MODEL_RF_SVM_ENSEMBLE;
    return PRE_CLASSIFICATION_MODEL_RF_SVM_ENSEMBLE; // default
}

}

