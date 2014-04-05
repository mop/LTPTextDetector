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
#ifndef SVMRFCONNECTEDCOMPONENTCLASSIFIER_H_
#define SVMRFCONNECTEDCOMPONENTCLASSIFIER_H_

#include <memory>

#include "ConnectedComponentClassifier.h"

namespace TextDetector {

class SVMRFConnectedComponentClassifier: public ConnectedComponentClassifier {
public:
	SVMRFConnectedComponentClassifier(
		const std::shared_ptr<ConnectedComponentClassifier> &clf_rf,
		const std::shared_ptr<ConnectedComponentClassifier> &clf_svm);
	virtual ~SVMRFConnectedComponentClassifier() = default;

	//! @see ConnectedComponentClassifier
    virtual void classify(
        const cv::Mat &f,
        const cv::Mat &cc_img,
        double &prob,
        std::vector<double> &v) override;
private:
	std::shared_ptr<ConnectedComponentClassifier> _classify_rf;
	std::shared_ptr<ConnectedComponentClassifier> _classify_svm;
};

} /* namespace TextDetector */

#endif /* SVMRFCONNECTEDCOMPONENTCLASSIFIER_H_ */
