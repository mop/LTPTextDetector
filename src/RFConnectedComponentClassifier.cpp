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
#include <text_detector/RFConnectedComponentClassifier.h>

namespace TextDetector {

RFConnectedComponentClassifier::RFConnectedComponentClassifier(
		const std::shared_ptr<cv::RandomTrees> &clf)
: _classifier(clf)
{
}

void RFConnectedComponentClassifier::classify(
        const cv::Mat &f,
        const cv::Mat &cc_img,
        double &prob,
        std::vector<double> &v)
{
	prob = _classifier->predict_prob(f);
	v.push_back(prob);
}

} /* namespace TextDetector */
