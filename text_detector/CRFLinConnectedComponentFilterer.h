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
#ifndef CRFLINCONNECTEDCOMPONENTFILTERER_H
#define CRFLINCONNECTEDCOMPONENTFILTERER_H

#include "ConnectedComponentFilterer.h"
#include "config.h"

#include <opencv2/core/core.hpp>

namespace TextDetector {

/**
 * A component filterer which filters the connected components with a linear conditional random field.
 */
class CRFLinConnectedComponentFilterer : public ConnectedComponentFilterer
{
public:
    CRFLinConnectedComponentFilterer(
        const cv::Mat &train_image,
        const cv::Mat &gradient_image, 
        const cv::Mat &unary_features,
        const dlib::graph_labeler<vector_type> &labeler
    );
    virtual ~CRFLinConnectedComponentFilterer();

    virtual std::vector<std::pair<int, std::vector<cv::Point> > >  filter_compontents(
        const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        std::vector<MserElement> &all_elements,
        const std::vector<std::vector<double> > &per_classifier_probs
    );

    void set_labeler(const dlib::graph_labeler<vector_type> &labeler) { _labeler = labeler; }
    dlib::graph_labeler<vector_type> get_labeler() const { return _labeler; }

    //void set_unary_features(const cv::Mat &unary) { _unary_features = unary; }
    const cv::Mat& get_unary_features() const { return _unary_features; }

    //void set_train_image(const cv::Mat &img) { _train_image = img; }
    const cv::Mat& get_train_image() const { return _train_image; }

    //void set_gradient_image(const cv::Mat &img) { _gradient_image = img; }
    const cv::Mat& get_gradient_image() const { return _gradient_image; }

private:
    //! Reference to the computed unary features
    const cv::Mat &_unary_features;
    //! Reference to the gradient image
    const cv::Mat &_gradient_image;
    //! Reference to the training image
    const cv::Mat &_train_image;
    
    //! graph labeller
    dlib::graph_labeler<vector_type> _labeler;
};

}

#endif /* end of include guard: CRFLINCONNECTEDCOMPONENTFILTERER_H */
