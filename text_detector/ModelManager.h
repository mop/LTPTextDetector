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

#ifndef MODELMANAGER_H

#define MODELMANAGER_H

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "LibSVMClassifier.h"
#include "ConfigurationManager.h"
#include "config.h"
#include "CNN.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <dlib/svm_threaded.h>
#pragma GCC diagnostic pop

namespace TextDetector {

class ModelManager
{
public:
    ModelManager(const std::shared_ptr<ConfigurationManager> &cfg);
    ~ModelManager() = default;

    dlib::graph_labeler<vector_type> get_graph_labeler() { return _labeler; }
    std::shared_ptr<CNN> get_unary_cnn_classifier() { return _cnn; }
    std::shared_ptr<cv::RandomTrees> get_unary_random_forest_classifier() { return _random_forest; }
    std::shared_ptr<cv::RandomTrees> get_pairwise_1_1_classifier() { return _pairwise_1_1_tree; }
    std::shared_ptr<cv::RandomTrees> get_pairwise_1_0_classifier() { return _pairwise_1_0_tree; }
    std::shared_ptr<cv::RandomTrees> get_pairwise_0_0_classifier() { return _pairwise_0_0_tree; }
    std::shared_ptr<LibSVMClassifier> get_svm_classifier() { return _svm_classifier; }
private:
    dlib::graph_labeler<vector_type> _labeler;
    std::shared_ptr<CNN> _cnn;
    std::shared_ptr<cv::RandomTrees> _random_forest;
    std::shared_ptr<LibSVMClassifier> _svm_classifier;
    std::shared_ptr<cv::RandomTrees> _pairwise_1_1_tree;
    std::shared_ptr<cv::RandomTrees> _pairwise_1_0_tree;
    std::shared_ptr<cv::RandomTrees> _pairwise_0_0_tree;
};

}

#endif /* end of include guard: MODELMANAGER_H */
