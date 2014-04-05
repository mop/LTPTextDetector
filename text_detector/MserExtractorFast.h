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

#ifndef MSEREXTRACTORFAST_H

#define MSEREXTRACTORFAST_H

#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <memory>
#include <utility>
#include <vector>

#include "CCUtils.h"
#include "ConnectedComponentExtractor.h"
#include "mser.h"

namespace TextDetector {
class ConnectedComponentClassifier;
} /* namespace TextDetector */

namespace TextDetector {

/**
 * This class is responsible for extracting MSER components from an image
 */
class MserExtractorFast : public ConnectedComponentExtractor
{
public:
    /**
     *  Constructs the mser extractor.
     *
     *  @param image_color is the color image from thich the components are extracted
     *  @param image_gray is the grayscale image on which the mser algorithm is applied
     *  @param gradient_image is the gradient image
     *  @param mask is the binarized response mask from the initial texture classifier
     *  @param model_manager is a reference to the ModelManager class, which holds our models
     *  @param uid_offset is an offset which is used for extracting components from 
     *         multiple image sources
     */
    MserExtractorFast(
        const cv::Mat &image_color,
        const cv::Mat &image_gray,
        const cv::Mat &gradient_image,
        const cv::Mat &mask,
        const std::shared_ptr<ConnectedComponentClassifier> &clf,
        int uid_offset
    );
    ~MserExtractorFast() = default;

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
        std::vector<MserElement> &all_elements
    ) override;

private:
    /**
     *  Returns the predictions from the classifiers for f and stores them in prob and v
     *
     *  @param f is the feature vector which should be predicted
     *  @param bwimg is the binary image of the sample
     *  @param prob (OUT) is the resulting (average) probability from each classifier model.
     *  @param v (OUT) is a vector which contains one probability for each used classifier.
     */
    inline void get_predictions(const cv::Mat &f, const cv::Mat &bwimg, double &prob, std::vector<double> &v);
	void create_uid_to_index_map(std::vector<MSER::Region> regions[2],
			std::unordered_map<int, int> uid_to_index[2]);
	void compute_features(const cv::Mat& swt1, const cv::Mat& swt2,
			MserElement& el);
    void compute_probs(
        size_t idx, const MserElement &el,
        std::vector<double> &probs,
        std::vector<std::vector<double> > &per_classifier_probs);

    //! Reference to the color image
    const cv::Mat &_image_color;
    //! Reference to the gray image
    const cv::Mat &_image_gray;
    //! Reference to the binary response mask
    const cv::Mat &_mask;
    //! Reference to the gradient image
    const cv::Mat &_gradient_image;
    //! Reference to the model manager
    std::shared_ptr<ConnectedComponentClassifier> _classifier;
    //! UID offset - used for multiple channels of images
    int _uid_offset;
};

}

#endif /* end of include guard: MSEREXTRACTOR_H */
