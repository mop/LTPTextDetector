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
#ifndef COMPONENTEXTRACTOR_H

#define COMPONENTEXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <utility>
#include <vector>

#include "CCUtils.h"

namespace TextDetector {

/**
 * This class is an interface for extracting connected components.
 */
class ConnectedComponentExtractor
{
public:
    ConnectedComponentExtractor() = default;
    virtual ~ConnectedComponentExtractor() = default;

    /**
     *  Extracts the MSER features and applies the appropriate classifier.
     *
     *  @param unary_features (OUT) is an empty matrix, which is filled by unary
     *                              features iff. ConfigurationManager::keep_unary_features
     *                              is TRUE
     *  @param comps (OUT) is a list of tuples. It stores an (UID, Pixel-List)
     *                     pair. The UID is the index into the
     *                     all_elements list, the probs list and the
     *                     per_classifier_probs list.
     *  @param probs (OUT) is a list of probabilities which is later used for grouping.
     *                     It contains either the average of several classifier results,
     *                     or the result of a single classifier
     *                     (depending on ConfigurationManager::pre_classification_model)
     *  @param per_classifier_probs (OUT) is a list of per classifier probabilities. Each
     *                     element contains a list of probabilities - one for each used
     *                     classifier
     *  @param all_elements is a list of MserElements, on which compute_features was called.
     */
    virtual void extract(
        cv::Mat &unary_features,
        std::vector<double> &probs,
        std::vector<std::vector<double> > &per_classifier_probs,
        std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        std::vector<MserElement> &all_elements) = 0;
protected:
    /**
     * Returns true if the given connected component is invalid. Otherwise
     * false is returned.
     *
     * @param mask is the detector mask
     * @param rect is the bounding box of the connected component
     */
    inline bool
    is_component_invalid(const cv::Mat &mask, const cv::Rect &rect) const;
};

bool ConnectedComponentExtractor::is_component_invalid(
	const cv::Mat &mask, const cv::Rect &rect) const
{
    float mask_sum = cv::sum(mask.colRange(rect.x, rect.x + rect.width
    	).rowRange(rect.y, rect.y + rect.height))[0];
    float rect_area = rect.width * rect.height * 255;
    bool below_min_overlap = mask_sum / rect_area < 0.3;
    return below_min_overlap || rect.width < 2 || rect.height < 5;
}
    
} /* namespace TextDetector */

#endif /* end of include guard: COMPONENTEXTRACTOR_H */
