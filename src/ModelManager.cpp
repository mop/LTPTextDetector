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
#include <text_detector/ModelManager.h>

namespace TextDetector {

ModelManager::ModelManager(const std::shared_ptr<ConfigurationManager> &mgr)
{
    if (mgr->get_svm_model_file() != "") {
        _svm_classifier.reset(
            new LibSVMClassifier(
                mgr->get_svm_model_file()));
    }

    if (mgr->get_cnn_model_file() != "") {
        _cnn.reset(
            new CNN(mgr->get_cnn_model_file()));
    }

    if (mgr->get_rf_model_file() != "") {
        _random_forest.reset(new cv::RandomTrees());
        cv::FileStorage fs(
            mgr->get_rf_model_file(),
            cv::FileStorage::READ);
        _random_forest->read(*fs, *fs["trees"]);
    }

    if (mgr->get_rf_model_pw_1_1_file() != "") {
        _pairwise_1_1_tree.reset(new cv::RandomTrees());
        cv::FileStorage fs(
            mgr->get_rf_model_pw_1_1_file(),
            cv::FileStorage::READ);
        _pairwise_1_1_tree->read(*fs, *fs["trees"]);
    }

    if (mgr->get_rf_model_pw_1_0_file() != "") {
        _pairwise_1_0_tree.reset(new cv::RandomTrees());
        cv::FileStorage fs(
            mgr->get_rf_model_pw_1_0_file(),
            cv::FileStorage::READ);
        _pairwise_1_0_tree->read(*fs, *fs["trees"]);
    }

    if (mgr->get_rf_model_pw_0_0_file() != "") {
        _pairwise_0_0_tree.reset(new cv::RandomTrees());
        cv::FileStorage fs(
            mgr->get_rf_model_pw_0_0_file(),
            cv::FileStorage::READ);
        _pairwise_0_0_tree->read(*fs, *fs["trees"]);
    }

    if (mgr->get_crf_model_file() != "") {
        std::ifstream ifs(mgr->get_crf_model_file().c_str(), std::ios::binary);
        dlib::deserialize(_labeler, ifs);
    }
}

}
