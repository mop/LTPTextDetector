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
#include <text_detector/RFConnectedComponentFilterer.h>

#include <algorithm>

namespace TextDetector {

RFConnectedComponentFilterer::~RFConnectedComponentFilterer() { }


std::vector<std::pair<int, std::vector<cv::Point> > >
RFConnectedComponentFilterer::filter_compontents(
    const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
    std::vector<MserElement> &all_elements,
    const std::vector<std::vector<double> > &per_classifier_probs)
{
    std::vector<std::pair<int, std::vector<cv::Point> > > results(comps);
    results.erase(
        std::remove_if(
            results.begin(), 
            results.end(), 
            [this, &per_classifier_probs] (const std::pair<int, std::vector<cv::Point> > &pair) -> bool {
        float sum = 0.0f;
        for (float p : per_classifier_probs[pair.first])
            sum += p;
        return sum / per_classifier_probs[pair.first].size() < _threshold;

        //return per_classifier_probs[pair.first][_index] < _threshold;
    }), results.end());
    return results;
}

}
