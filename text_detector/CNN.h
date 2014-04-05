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
#ifndef CNN_H
#define CNN_H

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
#include <memory>

namespace TextDetector {

class Layer 
{
public:
    virtual ~Layer() {}
    virtual cv::MatND fprop(const cv::MatND &input) const = 0;
};

//! This class is a ReLU convolutional layer.
//! It also max-pools the responses
class ConvLayer : public Layer
{
public:
    ConvLayer(const cv::MatND &weights, const cv::MatND &biases, int pad, int pool_size, int pool_stride);
    virtual ~ConvLayer();

    virtual cv::MatND fprop(const cv::MatND &input) const;
private:
    cv::MatND _weights;
    cv::MatND _biases;
    int _pad;
    int _pool_size;
    int _pool_stride;
};

class FullyConnectedLayer : public Layer
{
public:
    FullyConnectedLayer(const cv::Mat &weights, const cv::Mat &biases);
    virtual ~FullyConnectedLayer();
    virtual cv::MatND fprop(const cv::MatND &input) const;
private:
    cv::Mat _weights;
    cv::Mat _biases;
};

class SoftmaxLayer : public Layer 
{
public:
    SoftmaxLayer(const cv::Mat &weights, const cv::Mat &biases);
    virtual ~SoftmaxLayer() {}
    virtual cv::MatND fprop(const cv::MatND &input) const;
private:
    cv::Mat _weights;
    cv::Mat _biases;

};

//! This class is a simple implementation for a convolutional neural network. 
//! It only supports fprops.
class CNN 
{
public:
    CNN(const std::string &filename);
    ~CNN() {}

    cv::Mat fprop(const cv::Mat &input);
private:
    std::vector<std::unique_ptr<Layer> > _layers;
};

}


#endif /* end of include guard: CNN_H */
