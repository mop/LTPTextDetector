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
#include <text_detector/CNN.h>
#include <text_detector/cnpy.h>

#include <iostream>
#include <cstring>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <xmmintrin.h>
namespace TextDetector {

ConvLayer::ConvLayer(const cv::MatND &weights, const cv::MatND &biases, int pad, int pool_size, int pool_stride)
: _weights(weights), _biases(biases), _pad(pad), _pool_size(pool_size), _pool_stride(pool_stride)
{}

ConvLayer::~ConvLayer() {}

/**
 * This method applies the kernel on image-position (row,col,image_stack)
 * and stores the result in result at position (row, col, output_feature_map)
 */
static inline void
do_conv_mult(const cv::MatND &input, const cv::MatND &kernel, cv::MatND &result, int row, int col, int image_stack, int nfeature_maps, int pad)
{
    const int kernel_half_h = kernel.size[1]/2;
    const int kernel_half_w = kernel.size[2]/2;

    const float *kernel_ptr = (const float*)kernel.data;
    const float *image_ptr = (const float*) input.data;
    float *result_ptr = (float *)result.data;

    for (int i = -kernel_half_h; i <= kernel_half_h; i++) {
        const int y = row + i - pad;
        for (int j = -kernel_half_w; j <= kernel_half_w; j++) {
            const int x = col + j - pad;
            const float image_value = (y >= 0 && y < input.size[0] && x >= 0 && x < input.size[1]) ? 
                *(image_ptr + input.step[0]/4 * y + input.step[1]/4 * x + image_stack) :
                0.0f;
            //for (int output_feature_map = 0; output_feature_map < nfeature_maps; output_feature_map++) {
            //const __m128 scalar = _mm_set1_ps(image_value);
            for (int output_feature_map = 0; output_feature_map < nfeature_maps; output_feature_map++) {
                //int kernel_idx[] = {image_stack, i+kernel_half_h, j+kernel_half_w, output_feature_map};
                //const float image_value = input.at<float>(row+i, col+j, image_stack);
                //const float kernel_value = kernel.at<float>(kernel_idx);
                //result.at<float>(row - kernel_half_h, col - kernel_half_w, output_feature_map) += image_value * kernel_value;
                //__m128 kernvals = _mm_load_ps(kernel_ptr + kernel.step[0]/4 * image_stack + kernel.step[1]/4 * (i+kernel_half_h) + kernel.step[2]/4 * (j+kernel_half_w) +  output_feature_map);
                //__m128 resultvals = _mm_load_ps(result_ptr + (row - kernel_half_h) * result.step[0] / 4 + (col - kernel_half_w) * result.step[1]/4 + output_feature_map);

                //kernvals = _mm_mul_ps(kernvals, scalar);
                //resultvals = _mm_add_ps(resultvals, kernvals);
                //_mm_store_ps(result_ptr + (row - kernel_half_h) * result.step[0] / 4 + (col - kernel_half_w) * result.step[1]/4 + output_feature_map, resultvals);

                const float kernel_value = *(kernel_ptr + kernel.step[0]/4 * image_stack + kernel.step[1]/4 * (i+kernel_half_h) + kernel.step[2]/4 * (j+kernel_half_w) +  output_feature_map);
                *(result_ptr + (row - kernel_half_h) * result.step[0] / 4 + (col - kernel_half_w) * result.step[1]/4 + output_feature_map) += image_value * kernel_value;
            }
        }
    }
}

cv::MatND ConvLayer::fprop(const cv::MatND &input) const
{

    // input is b,0,1,c where b is always '1'

    const int kernel_h_half = _weights.size[1]/2;
    const int kernel_w_half = _weights.size[2]/2;
    const int image_max_h = input.size[0] + 2*_pad - _weights.size[1]/2;
    const int image_max_w = input.size[1] + 2*_pad - _weights.size[2]/2;
    const int n_stacks = (input.dims > 2 ? input.size[2] : 1);
    // weights have shape (image, kernel-rows, kernel-cols, output-feature-maps)
    int sizes[] = { input.size[0] - _weights.size[1] + 1 + 2*_pad, input.size[1] - _weights.size[2] + 1 + 2*_pad, _weights.size[3] };
    cv::MatND result(3, sizes, CV_32FC1, cv::Scalar(0.0f));
    // each of our results consists of several convolutions from the inputs
    
    //for (int i = 0; i < result.size[2]; i++) {
        
        // foreach row
        #pragma omp parallel for
        for (int row = kernel_h_half; row < image_max_h; row++) {
            if (row < 0) continue;
            // foreach col
            for (int col = kernel_w_half; col < image_max_w; col++) {
                if (col < 0) continue;
                // foreach stack
                for (int stack = 0; stack < n_stacks; stack++) {
                    do_conv_mult(input, _weights, result, row, col, stack, result.size[2], _pad);
                }
            }
        }
    //}

    //std::cout << "convolution" << std::endl;
    //std::cout << result.size[0] << " " << result.size[1] << " " << result.size[2] << std::endl;
    result += _biases;
    float *result_ptr_wr = (float *) result.data;
    const int dim = result.size[0] * result.size[1] * result.size[2];
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        result_ptr_wr[i] = std::max(0.0f, result_ptr_wr[i]);
    }
    //std::cout << "activation" << std::endl;

    int size_pooled[] = {result.size[0] / _pool_stride, result.size[1] / _pool_stride, result.size[2]};
    cv::MatND pooled_result(3, size_pooled, CV_32FC1, cv::Scalar(0.0f));
    
    // max-pool over region

    float *pooled_result_ptr = (float*) pooled_result.data;
    const float *result_ptr = (const float *) result.data;

    #pragma omp parallel for 
    for (int row = 0; row < pooled_result.size[0]; row++) {
        int i = row * _pool_stride;
        for (int j = 0, col = 0; col < pooled_result.size[1]; j += _pool_stride, col++) {
            float maxvals[result.size[2]];
            memset(maxvals, 0.0f, sizeof (float) * result.size[2]);
            int sums = 0;
            for (int dy = 0; dy < _pool_size; dy++) {
                for (int dx = 0; dx < _pool_size; dx++) {
                    if (i+dy >= result.size[0] || j+dx >= result.size[1]) continue;
                    sums++;
                    for (int stack = 0; stack < result.size[2]; stack++) {
                        //maxvals[stack] = std::max(maxvals[stack], 
                        //    *(result_ptr + (i+dy) * result.step[0]/4 + (j+dx) * result.step[1]/4 + stack));
                        maxvals[stack] += 
                            *(result_ptr + (i+dy) * result.step[0]/4 + (j+dx) * result.step[1]/4 + stack);
                    }
                }
            }
            for (int stack = 0; stack < result.size[2]; stack++)
                *(pooled_result_ptr + row * pooled_result.step[0]/4 + col * pooled_result.step[1]/4 + stack) = maxvals[stack] / sums;
        }
    }
    //for (int i = 0; i < pooled_result.size[0]; i++) {
    //    for (int j = 0; j < pooled_result.size[0]; j++) {
    //        std::cout << pooled_result.at<float>(i,j,0) << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << "pooled" << std::endl;
    return pooled_result;
}


FullyConnectedLayer::FullyConnectedLayer(const cv::Mat &weights, const cv::Mat &biases)
: _weights(weights), _biases(biases)
{
}

FullyConnectedLayer::~FullyConnectedLayer(){}

cv::MatND FullyConnectedLayer::fprop(const cv::MatND &input) const
{
    cv::Mat flattened_inputs;
    if (input.dims > 1) {
        flattened_inputs = cv::Mat(1, input.size[0] * input.size[1] * input.size[2], CV_32FC1);
        for (int i = 0; i < flattened_inputs.cols; i++) {
            flattened_inputs.at<float>(i) = input.at<float>(i);
        }
    } else {
        flattened_inputs = input;
    }

    cv::Mat result = flattened_inputs * _weights + _biases.reshape(0, 1);
    cv::max(result, 0, result);
    return cv::MatND(result);
}

static cv::MatND to_cv(const cnpy::NpyArray &ary)
{
    int ndims = 1;
    std::vector<int> dims;
    for (int i = 0; i < ary.shape.size(); i++) {
        ndims *= ary.shape[i];
        dims.push_back(ary.shape[i]);
    }

    cv::MatND result(ary.shape.size(), &dims[0], CV_32FC1, cv::Scalar(0.0f));
    memcpy(result.data, ary.data, ndims * sizeof (float));
    return result;
}

CNN::CNN(const std::string &filename)
{
    cv::MatND c0_b = to_cv(cnpy::npy_load(filename + "/biases-l0.npy"));
    cv::MatND c0_w = to_cv(cnpy::npy_load(filename + "/weights-l0.npy"));
    cv::MatND c1_b = to_cv(cnpy::npy_load(filename + "/biases-l1.npy"));
    cv::MatND c1_w = to_cv(cnpy::npy_load(filename + "/weights-l1.npy"));
    cv::Mat fc_b = cv::Mat(to_cv(cnpy::npy_load(filename + "/biases-l2.npy")));
    cv::Mat fc_w = cv::Mat(to_cv(cnpy::npy_load(filename + "/weights-l2.npy")));
    cv::Mat svm_b = cv::Mat(to_cv(cnpy::npy_load(filename + "/biases-l3.npy")));
    cv::Mat svm_w = cv::Mat(to_cv(cnpy::npy_load(filename + "/weights-l3.npy")));

    _layers.push_back(std::unique_ptr<Layer>(new ConvLayer(c0_w, c0_b, 2, 3, 2)));
    _layers.push_back(std::unique_ptr<Layer>(new ConvLayer(c1_w, c1_b, 2, 3, 2)));
    _layers.push_back(std::unique_ptr<Layer>(new FullyConnectedLayer(fc_w, fc_b)));
    _layers.push_back(std::unique_ptr<Layer>(new SoftmaxLayer(svm_w, svm_b)));
}

SoftmaxLayer::SoftmaxLayer(const cv::Mat &weights, const cv::Mat &biases)
: _weights(weights), _biases(biases)
{
}

cv::MatND SoftmaxLayer::fprop(const cv::MatND &input) const
{
    cv::Mat flattened_inputs;
    if (input.dims > 2) {
        flattened_inputs = cv::Mat(1, input.size[0] * input.size[1] * input.size[2], CV_32FC1);
        for (int i = 0; i < flattened_inputs.cols; i++) {
            flattened_inputs.at<float>(i) = input.at<float>(i);
        }
    } else {
        flattened_inputs = input;
    }

    cv::Mat result = flattened_inputs * _weights + _biases.reshape(1,1);
    cv::min(result, 50, result);
    cv::max(result, -50, result);
    cv::exp(result, result);
    float norm = cv::sum(result)[0]; 
    result = result / (norm + 1e-10);
    return cv::MatND(result);
}


cv::Mat CNN::fprop(const cv::Mat &input)
{
    cv::MatND mat(input);
    int sizes[] = {28,28,1};
    for (int i = 0; i < _layers.size(); i++) {
        mat = _layers[i]->fprop(mat);
    }
    return cv::Mat(mat);
}

}


//int main(int argc, const char *argv[])
//{
//    if (argc < 3) {
//        std::cerr << "SCREWUP" << std::endl;
//    }
//    CNN cnn(argv[1]);
//    for (int i = 0; i < 1000; i++) {
//        cnn.fprop(cv::Mat::ones(28,28, CV_32FC1));
//    }
//    //return 0;
//    std::cout << cnn.fprop(cv::Mat::ones(28,28, CV_32FC1)) << std::endl;
//    CvMLData data;
//    data.set_delimiter(',');
//    data.read_csv(argv[2]);
//    data.set_response_idx(0);
//
//    cv::Mat dat = data.get_values();
//    int errs = 0;
//    for (int i = 0; i < dat.rows; i++) {
//        cv::Mat m = dat.row(i).colRange(1, dat.cols);
//        cv::Mat responses = cnn.fprop(m.reshape(1,28) / 255.0f);
//        float lbl = responses.at<float>(0) > responses.at<float>(1) ? -1 : 1;
//        if (lbl != dat.at<float>(i,0)) {
//            std::cout << "GT: " << dat.at<float>(i,0) << " PRED: " << lbl << std::endl;
//            cv::imshow("IMG", m.reshape(1,28) / 255.0);
//            cv::waitKey(0);
//            errs++;
//        }
//        if (i % 100 == 0) {
//            std::cout << i << "/" << dat.rows << std::endl;
//            std::cout << "ERR: " << float(errs) / i << std::endl;
//        }
//    }
//    std::cout << float(errs) / dat.rows << std::endl;
//    std::cout << dat.rows << " " << dat.cols << std::endl;
//
//    return 0;
//}
