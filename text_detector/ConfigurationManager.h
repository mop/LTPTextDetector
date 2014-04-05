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
#ifndef CONFIGURATIONMANAGER_H

#define CONFIGURATIONMANAGER_H

#include <string>
#include <memory>

namespace TextDetector {

/**
 * This class is responsible for loading the configuration from a yaml file
 */
class ConfigurationManager 
{
public:
    ConfigurationManager(const std::string &filename);
    ~ConfigurationManager() {}

    enum PreClassificationModel {
        PRE_CLASSIFICATION_MODEL_RANDOM_FOREST = 1,
        PRE_CLASSIFICATION_MODEL_SVM,
        PRE_CLASSIFICATION_MODEL_CNN,
        PRE_CLASSIFICATION_MODEL_RF_SVM_ENSEMBLE
    };
    enum ClassificationModel { 
        CLASSIFICATION_MODEL_RANDOM_FOREST = 1,
        CLASSIFICATION_MODEL_CRF_LIN,
        CLASSIFICATION_MODEL_CRF_RF
    };

    enum WordSplitModel { 
        MODEL_PROJECTION_PROFILE = 1,
        MODEL_PROJECTION_PROFILE_SOFT,
        MODEL_SIMPLE
    };

    enum FeaturePoolSingular {
        POOL_DEFAULT = 1,
        POOL_DEFAULT_HOG
    };

    //! Returns the model used for splitting words
    int get_word_split_model() const { return _word_split_model; }
    //! Returns the model used for classification
    int get_classification_model() const { return _classification_model; }
    //! Returns the model used for pre-classification
    int get_preclassification_model() const { return _pre_classification_model; }

    std::string get_config_filename() const { return _config_filename; }

    //! Returns the directory in which all input-files (i.e. the test-images) are stored
    std::string get_input_directory() const { return _input_directory; }
    //! Returns the directory in which all responses are stored
    std::string get_responses_directory() const { return _responses_directory; }
    //! Returns the directory in which the CRF is stored
    std::string get_crf_model_file() const { return _crf_model_file; }
    //! Returns the directory in which the SVM is stored
    std::string get_svm_model_file() const { return _svm_model_file; }
    //! Returns the directory in which the CNN is stored
    std::string get_cnn_model_file() const { return _cnn_model_file; }
    //! Returns the directory in which the Random Forest is stored
    std::string get_rf_model_file() const { return _random_forest_model_file; }

    //! Returns the model file for the pairwise random forest
    std::string get_rf_model_pw_1_1_file() const { return _random_forest_pw_1_1_model_file; }
    //! Returns the model file for the pairwise random forest
    std::string get_rf_model_pw_1_0_file() const { return _random_forest_pw_1_0_model_file; }
    //! Returns the model file for the pairwise random forest
    std::string get_rf_model_pw_0_0_file() const { return _random_forest_pw_0_0_model_file; }

    //! Returns the single feature pool used
    int get_feature_pool_singular() const { return _feature_pool_singular; }
    //! Returns the pairwise feature pool used
    int get_feature_pool_pairwise() const { return _feature_pool_pairwise; }
    //! Returns true if a cache is available
    bool has_cache() const { return _cache_dir != ""; }
    //! Returns true if the svm grouping stage should be ignored
    bool ignore_grouping_svm() const { return _ignore_grouping_svm; }
    //! Returns the cached probabs
    std::string get_cache_directory() { return _cache_dir; }


    //! Returns true if verbose is on
    bool verbose() const { return _verbose; }
    
    //! Returns the random seed
    int get_random_seed() const { return _random_seed; }

    //! Returns the threshold used for the classifier masks
    float get_threshold() const { return _threshold; }
    //! Returns the threshold for the RFConnectedComponentFilterer
    float get_pre_classification_prob_threshold() const { return _pre_classification_prob_threshold; }
    //! Returns the threshold used for the word groups
    float get_word_group_threshold() const { return _word_group_threshold; }
    //! Returns the maximum height ratio
    float get_maximum_height_ratio() const { return _maximum_height_ratio; }
    //! Returns the minimum overlap height ratio
    float get_minimum_vertical_overlap() const { return _minimum_vertical_overlap; }
    //! Returns true if we allow single letter splits
    bool allow_single_letters() const { return _allow_single_letters; }

    //! Returns true if the unary matrix should actually be filled with features. This is necessary for the
    //! linear crf, since other models just reuse the probabilities
    bool keep_unary_features() const { return _classification_model == CLASSIFICATION_MODEL_CRF_LIN; }
    //! Returns true if responses should be used
    bool ignore_responses() const { return _ignore_responses; }

    //! Sets the global configuration instance
    static void set_instance(std::shared_ptr<ConfigurationManager> instance) { ConfigurationManager::_instance = instance; }
    //! Retuns the global configuration instance
    static std::shared_ptr<ConfigurationManager> instance() { return ConfigurationManager::_instance; }

    //! Returns true if color channels should be excluded
    bool ignore_color() const { return _ignore_color; }
    //! Returns true if the graychannel should be ignored
    bool ignore_gray() const { return _ignore_gray; }

    //! Return true if word splitting should not be done
    bool ignore_word_splitting() const { return _ignore_word_splitting; }

    //! Returns true if GT mask segmentation should be used
    bool include_binary_masks() const { return _include_binary_masks; }
    //! Get the minimum grouping size
    int get_min_group_size() const { return _min_group_size; }

    bool get_set_gt_prop_to_one() const { return _set_gt_prop_to_one; }
private:
    static std::shared_ptr<ConfigurationManager> _instance;

    int get_word_split_model(const std::string &key) const;
    int get_classification_model(const std::string &key) const;
    int get_pre_classification_model(const std::string &key) const;

    std::string _config_filename;

    std::string _input_directory;
    std::string _responses_directory;
    std::string _crf_model_file;
    std::string _svm_model_file;
    std::string _cnn_model_file;
    std::string _random_forest_model_file;

    std::string _random_forest_pw_1_1_model_file;
    std::string _random_forest_pw_1_0_model_file;
    std::string _random_forest_pw_0_0_model_file;

    int _word_split_model;
    int _classification_model;
    int _pre_classification_model;
    int _random_seed;
    int _min_group_size;

    int _feature_pool_singular;
    int _feature_pool_pairwise;

    float _threshold;
    //! Threshold used for word grouping. Groups of words having 
    //! an average probability below this threshold are not considered as valid words.
    float _word_group_threshold;
    float _pre_classification_prob_threshold;
    bool _allow_single_letters;
    float _minimum_vertical_overlap;
    float _maximum_height_ratio;
    float _significance_threshold;

    bool _verbose;
    bool _ignore_responses; 
    bool _ignore_gray;
    bool _ignore_color;
    bool _ignore_word_splitting;
    bool _include_binary_masks;
    bool _ignore_grouping_svm;
    bool _set_gt_prop_to_one;
    std::string _cache_dir;
};

}

#endif /* end of include guard: CONFIGURATIONMANAGER_H */
