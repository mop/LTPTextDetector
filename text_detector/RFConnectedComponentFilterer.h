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

#ifndef RFCONNECTEDCOMPONENTFILTERER_H
#define RFCONNECTEDCOMPONENTFILTERER_H

#include "ConnectedComponentFilterer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

namespace TextDetector {

/**
 * This class filters connected components based on a (random forest) classifier response.
 */
class RFConnectedComponentFilterer : public ConnectedComponentFilterer
{
public:
    /**
     *  Initializes the model with an index into the list of per_classifier_probs and a threshold.
     *
     *  @param prob_idx is the index of the classifier in the per_classifier_probs vectors. If
     *         i.e. prob_idx is 1, then per_classifier_probs[i][1] is used to threshold the CCs
     *  @param threshold is the threshold which is used to filter components
     */
    RFConnectedComponentFilterer(int prob_idx = 0, float t = 0.20): _index(prob_idx), _threshold(t) {}
    virtual ~RFConnectedComponentFilterer();

    virtual std::vector<std::pair<int, std::vector<cv::Point> > >  filter_compontents(
        const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        std::vector<MserElement> &all_elements,
        const std::vector<std::vector<double> > &per_classifier_probs
    );

    void set_threshold(float t) { _threshold = t; }
    float get_threshold() const { return _threshold; }
    void set_index(int i) { _index = i; }
    int get_index() const { return _index; }
private:
    float _threshold;
    int _index;
};

}

#endif /* end of include guard: RFCONNECTEDCOMPONENTFILTERER_H */
