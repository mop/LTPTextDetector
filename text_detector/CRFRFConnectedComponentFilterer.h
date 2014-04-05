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
#ifndef CRFRFCONNECTEDCOMPONENTFILTERER_H
#define CRFRFCONNECTEDCOMPONENTFILTERER_H

#include "ConnectedComponentFilterer.h"

namespace TextDetector {

/**
 * This class filters the connected components with a conditional random fields
 * using random forest probabilities as unary and pairwise potentials.
 */
class CRFRFConnectedComponentFilterer : public ConnectedComponentFilterer 
{
public:
    /**
     *  Constructs the filterer with the given training image,
     *  a gradient image, a graph labeler used for inference
     *  and 3 classifier for computing the pairwise potentials
     */
    CRFRFConnectedComponentFilterer(
        const cv::Mat &train_image, 
        const cv::Mat &gradient_image,
        const dlib::graph_labeler<vector_type> &labeler, 
        std::shared_ptr<cv::RandomTrees> pairwise_1_1_tree,
        std::shared_ptr<cv::RandomTrees> pairwise_1_0_tree,
        std::shared_ptr<cv::RandomTrees> pairwise_0_0_tree
    );
    virtual ~CRFRFConnectedComponentFilterer();

    /**
     *  Filters the given list of components.
     *
     *  @param comps is a list of tuples (index, component), where index is the 
     *               index into the all_elements and per_classifier_probs vector
     *               and component is a list of pixel-coordinates
     *  @param all_elements is a list of all elements ever extracted
     *  @param per_classifier_probs is a list of probabilities for the unary potentials.
     *         This is already computed in a previous step, so we just re-use the 
     *         cached values here
     */
    virtual std::vector<std::pair<int, std::vector<cv::Point> > >  
    filter_compontents(
        const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        std::vector<MserElement> &all_elements,
        const std::vector<std::vector<double> > &per_classifier_probs
    );

private:
    const cv::Mat& _train_image;
    const cv::Mat& _gradient_image;

    dlib::graph_labeler<vector_type> _labeler;

    std::shared_ptr<cv::RandomTrees> _pairwise_1_1_tree;
    std::shared_ptr<cv::RandomTrees> _pairwise_1_0_tree;
    std::shared_ptr<cv::RandomTrees> _pairwise_0_0_tree;
};

}

#endif /* end of include guard: CRFRFCONNECTEDCOMPONENTFILTERER_H */
