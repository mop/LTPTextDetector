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
#ifndef SVMCONNECTEDCOMPONENTCLASSIFIER_H_
#define SVMCONNECTEDCOMPONENTCLASSIFIER_H_

#include <opencv2/core/core.hpp>
#include <memory>
#include <vector>

#include "ConnectedComponentClassifier.h"
#include "LibSVMClassifier.h"

namespace TextDetector {
/**
 * This class uses an SVM classifier for connected component classification
 */
class SVMConnectedComponentClassifier : public ConnectedComponentClassifier {
public:
	SVMConnectedComponentClassifier(const std::shared_ptr<LibSVMClassifier> &cls);
	virtual ~SVMConnectedComponentClassifier() = default;

	//! @see ConnectedComponentClassifier
    virtual void classify(
        const cv::Mat &f,
        const cv::Mat &cc_img,
        double &prob,
        std::vector<double> &v) override;
private:
    //! Holds a classifier
    std::shared_ptr<LibSVMClassifier> _classifier;
};
}

#endif /* SVMCONNECTEDCOMPONENTCLASSIFIER_H_ */
