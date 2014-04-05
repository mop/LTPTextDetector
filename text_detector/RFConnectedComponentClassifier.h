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


#ifndef RFCONNECTEDCOMPONENTCLASSIFIER_H_
#define RFCONNECTEDCOMPONENTCLASSIFIER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <memory>
#include <vector>

#include "ConnectedComponentClassifier.h"

namespace TextDetector {

class RFConnectedComponentClassifier: public ConnectedComponentClassifier {
public:
	RFConnectedComponentClassifier(const std::shared_ptr<cv::RandomTrees> &clf);
	virtual ~RFConnectedComponentClassifier() = default;

	//! @see ConnectedComponentClassifier
    virtual void classify(
        const cv::Mat &f,
        const cv::Mat &cc_img,
        double &prob,
        std::vector<double> &v);
private:
    std::shared_ptr<cv::RandomTrees> _classifier;
};

} /* namespace TextDetector */

#endif /* RFCONNECTEDCOMPONENTCLASSIFIER_H_ */
