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
#ifndef CONNECTEDCOMPONENTCLASSIFIER_H_
#define CONNECTEDCOMPONENTCLASSIFIER_H_

#include <opencv2/core/core.hpp>
#include <vector>

namespace TextDetector {

/**
 * This class is responsible for classifying connected components.
 */
class ConnectedComponentClassifier
{
public:
	ConnectedComponentClassifier() = default;
	virtual ~ConnectedComponentClassifier() = default;

	/**
	 * Classifies a connected component
	 * @param f [IN] is the feature vector
	 * @param cc_img [IN] is an image of the connected component
	 * @param prob [OUT] is a probability with which
	 * 			the component is a text-component
	 * @param v [OUT] is a list of probabilities from each discriminative
	 * 				  model
	 */
    virtual void classify(
        const cv::Mat &f,
        const cv::Mat &cc_img,
        double &prob,
        std::vector<double> &v) = 0;
};

} /* namespace TextDetector */

#endif /* CONNECTEDCOMPONENTCLASSIFIER_H_ */
