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
#include <text_detector/SVMRFConnectedComponentClassifier.h>

namespace TextDetector {

SVMRFConnectedComponentClassifier::SVMRFConnectedComponentClassifier(
		const std::shared_ptr<ConnectedComponentClassifier> &clf_rf,
		const std::shared_ptr<ConnectedComponentClassifier> &clf_svm)
: _classify_rf(clf_rf), _classify_svm(clf_svm)
{
}

void SVMRFConnectedComponentClassifier::classify(
    const cv::Mat &f,
    const cv::Mat &cc_img,
    double &prob,
    std::vector<double> &v)
{
	double p1 = 0.0, p2 = 0.0;
	_classify_rf->classify(f, cc_img, p1, v);
	_classify_svm->classify(f, cc_img, p2, v);
	prob = 0.5 * p1 + 0.5 * p2;
}
} /* namespace TextDetector */
