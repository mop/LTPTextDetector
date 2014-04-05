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
#include <text_detector/LTPComputer.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace TextDetector {


LTPComputer::LTPComputer(int maps)
: _nmaps(maps) {}

static cv::Mat gy(const cv::Mat &rgb_image)
{
    cv::Mat result(rgb_image.rows, rgb_image.cols, CV_32FC1, cv::Scalar(0.0f));
    const unsigned char* ptr = (const unsigned char*)rgb_image.data;
    float *result_ptr = (float*) result.data;
    #pragma omp parallel for
    for (int i = 0; i < rgb_image.rows; i++) {
        for (int j = 0; j < rgb_image.cols; j++) {
            const float r1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 0) / 255.0f;
            const float g1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 1) / 255.0f;
            const float b1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 2) / 255.0f;

            const float r2 = (i+1) >= rgb_image.rows ? r1 : (*(ptr+(i+1) * rgb_image.step[0] + j * rgb_image.step[1] + 0) / 255.0f);
            const float g2 = (i+1) >= rgb_image.rows ? g1 : (*(ptr+(i+1) * rgb_image.step[0] + j * rgb_image.step[1] + 1) / 255.0f);
            const float b2 = (i+1) >= rgb_image.rows ? b1 : (*(ptr+(i+1) * rgb_image.step[0] + j * rgb_image.step[1] + 2) / 255.0f);

            float max = std::abs(r1 - r2);
            float maxval = r1 - r2;
            if (std::abs(g1 - g2) > max) {
                max = std::abs(g1 - g2);
                maxval = g1 - g2;
            }
            if (std::abs(b1 - b2) > max) {
                max = std::abs(b1 - b2);
                maxval = b1 - b2;
            }
            *(result_ptr + i * result.step[0]/4 + j * result.step[1]/4) = maxval;
        }
    }
    return result;
}

static cv::Mat gx(const cv::Mat &rgb_image)
{
    cv::Mat result(rgb_image.rows, rgb_image.cols, CV_32FC1, cv::Scalar(0.0f));
    const unsigned char* ptr = (const unsigned char*)rgb_image.data;
    float *result_ptr = (float*) result.data;
    #pragma omp parallel for
    for (int i = 0; i < rgb_image.rows; i++) {
        for (int j = 0; j < rgb_image.cols; j++) {
            const float r1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 0) / 255.0f;
            const float g1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 1) / 255.0f;
            const float b1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 2) / 255.0f;

            const float r2 = (j+1) >= rgb_image.cols ? r1 : (*(ptr+i * rgb_image.step[0] + (j+1) * rgb_image.step[1] + 0) / 255.0f);
            const float g2 = (j+1) >= rgb_image.cols ? g1 : (*(ptr+i * rgb_image.step[0] + (j+1) * rgb_image.step[1] + 1) / 255.0f);
            const float b2 = (j+1) >= rgb_image.cols ? b1 : (*(ptr+i * rgb_image.step[0] + (j+1) * rgb_image.step[1] + 2) / 255.0f);

            float max = std::abs(r1 - r2);
            float maxval = r1 - r2;
            if (std::abs(g1 - g2) > max) {
                max = std::abs(g1 - g2);
                maxval = g1 - g2;
            }
            if (std::abs(b1 - b2) > max) {
                max = std::abs(b1 - b2);
                maxval = b1 - b2;
            }
            *(result_ptr + i * result.step[0]/4 + j * result.step[1]/4) = maxval;
        }
    }
    return result;
}

static cv::Mat gad(const cv::Mat &rgb_image) 
{
    cv::Mat result(rgb_image.rows, rgb_image.cols, CV_32FC1, cv::Scalar(0.0f));
    const unsigned char* ptr = (const unsigned char*)rgb_image.data;
    float *result_ptr = (float*) result.data;
    #pragma omp parallel for
    for (int i = 0; i < rgb_image.rows; i++) {
        for (int j = 0; j < rgb_image.cols; j++) {
            // center (1)
            const float r1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 0) / 255.0f;
            const float g1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 1) / 255.0f;
            const float b1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 2) / 255.0f;

            // diagonal to upper right
            const float r2 = *(ptr+(MAX(0, i-1)) * rgb_image.step[0] + MIN(j+1, rgb_image.cols-1) * rgb_image.step[1] + 0) / 255.0f;
            const float g2 = *(ptr+(MAX(0, i-1)) * rgb_image.step[0] + MIN(j+1, rgb_image.cols-1) * rgb_image.step[1] + 1) / 255.0f;
            const float b2 = *(ptr+(MAX(0, i-1)) * rgb_image.step[0] + MIN(j+1, rgb_image.cols-1) * rgb_image.step[1] + 2) / 255.0f;

            float max = std::abs(r1 - r2);
            float maxval = r1 - r2;
            if (std::abs(g1 - g2) > max) {
                max = std::abs(g1 - g2);
                maxval = g1 - g2;
            }
            if (std::abs(b1 - b2) > max) {
                max = std::abs(b1 - b2);
                maxval = b1 - b2;
            }
            *(result_ptr + i * result.step[0]/4 + j * result.step[1]/4) = maxval;
        }
    }
    return result;
}

static cv::Mat gd(const cv::Mat &rgb_image) 
{
    cv::Mat result(rgb_image.rows, rgb_image.cols, CV_32FC1, cv::Scalar(0.0f));
    const unsigned char* ptr = (const unsigned char*)rgb_image.data;
    float *result_ptr = (float*) result.data;
    #pragma omp parallel for
    for (int i = 0; i < rgb_image.rows; i++) {
        for (int j = 0; j < rgb_image.cols; j++) {
            // center (1)
            const float r1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 0) / 255.0f;
            const float g1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 1) / 255.0f;
            const float b1 = *(ptr + i * rgb_image.step[0] + j * rgb_image.step[1] + 2) / 255.0f;

            // diagonal to the upper left (-1)
            const float r2 = *(ptr+(MAX(0, i-1)) * rgb_image.step[0] + MIN(j-1, rgb_image.cols - 1) * rgb_image.step[1] + 0) / 255.0f;
            const float g2 = *(ptr+(MAX(0, i-1)) * rgb_image.step[0] + MIN(j-1, rgb_image.cols - 1) * rgb_image.step[1] + 1) / 255.0f;
            const float b2 = *(ptr+(MAX(0, i-1)) * rgb_image.step[0] + MIN(j-1, rgb_image.cols - 1) * rgb_image.step[1] + 2) / 255.0f;

            float max = std::abs(r1 - r2);
            float maxval = r1 - r2;
            if (std::abs(g1 - g2) > max) {
                max = std::abs(g1 - g2);
                maxval = g1 - g2;
            }
            if (std::abs(b1 - b2) > max) {
                max = std::abs(b1 - b2);
                maxval = b1 - b2;
            }
            *(result_ptr + i * result.step[0]/4 + j * result.step[1]/4) = maxval;
        }
    }
    return result;
}

static void
make_ltp(
    cv::Mat &result,
    int l,
    const cv::Mat &e_x,
    const cv::Mat &e_y,
    const cv::Mat &e_d,
    const cv::Mat &e_ad,
    const cv::Mat &grad_x, 
    const cv::Mat &grad_y, 
    const cv::Mat &grad_d, 
    const cv::Mat &grad_ad)
{
    unsigned char *ptr = (unsigned char *) result.data;

    const float *e_x_ptr  = (const float *) e_x.data;
    const float *e_y_ptr  = (const float *) e_y.data;
    const float *e_d_ptr  = (const float *) e_d.data;
    const float *e_ad_ptr = (const float *) e_ad.data;

    const float *grad_x_ptr  = (const float *) grad_x.data;
    const float *grad_y_ptr  = (const float *) grad_y.data;
    const float *grad_d_ptr  = (const float *) grad_d.data;
    const float *grad_ad_ptr = (const float *) grad_ad.data;

    #pragma omp parallel for
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            {
            bool right    = *(grad_x_ptr  + i * grad_x.step[0]/4  + j * grad_x.step[1]/4)  > *(e_x_ptr  + i * e_x.step[0]/4  + j * e_x.step[1]/4);
            bool down     = *(grad_y_ptr  + i * grad_y.step[0]/4  + j * grad_y.step[1]/4)  > *(e_y_ptr  + i * e_y.step[0]/4  + j * e_y.step[1]/4);
            bool upleft   = *(grad_d_ptr  + i * grad_d.step[0]/4  + j * grad_d.step[1]/4)  > *(e_d_ptr  + i * e_d.step[0]/4  + j * e_d.step[1]/4);
            bool upright  = *(grad_ad_ptr + i * grad_ad.step[0]/4 + j * grad_ad.step[1]/4) > *(e_ad_ptr + i * e_ad.step[0]/4 + j * e_ad.step[1]/4);

            bool left      = (j == 0 ? 0.0f : -*(grad_x_ptr  +     i * grad_x.step[0]/4  + (j-1) * grad_x.step[1]/4))  > *(e_x_ptr  + i * e_x.step[0]/4  + j * e_x.step[1]/4);
            bool up        = (i == 0 ? 0.0f : -*(grad_y_ptr  + (i-1) * grad_y.step[0]/4  +     j * grad_y.step[1]/4))  > *(e_y_ptr  + i * e_y.step[0]/4  + j * e_y.step[1]/4);
            bool downright = (i+1 >= result.rows || j+1 >= result.cols) ? 0.0f : 
                        -*(grad_d_ptr  + (i+1) * grad_d.step[0]/4  + (j+1) * grad_d.step[1]/4)  > *(e_d_ptr  + i * e_d.step[0]/4  + j * e_d.step[1]/4);
            bool downleft = (i+1 >= result.rows || j == 0) ? 0.0f : 
                        -*(grad_ad_ptr + (i+1) * grad_ad.step[0]/4 + (j-1) * grad_ad.step[1]/4) > *(e_ad_ptr + i * e_ad.step[0]/4 + j * e_ad.step[1]/4);


            unsigned char lbp = down + upleft * 2 + right * 4 + upright * 8 + up * 16 + downright * 32 + left * 64 + downleft * 128;
            *(ptr+result.step[0] * i + result.step[1] * j + l*2) = lbp;
            }
            {
            bool right    = *(grad_x_ptr  + i * grad_x.step[0]/4  + j * grad_x.step[1]/4)  < -*(e_x_ptr  + i * e_x.step[0]/4  + j * e_x.step[1]/4);
            bool down     = *(grad_y_ptr  + i * grad_y.step[0]/4  + j * grad_y.step[1]/4)  < -*(e_y_ptr  + i * e_y.step[0]/4  + j * e_y.step[1]/4);
            bool upleft   = *(grad_d_ptr  + i * grad_d.step[0]/4  + j * grad_d.step[1]/4)  < -*(e_d_ptr  + i * e_d.step[0]/4  + j * e_d.step[1]/4);
            bool upright  = *(grad_ad_ptr + i * grad_ad.step[0]/4 + j * grad_ad.step[1]/4) < -*(e_ad_ptr + i * e_ad.step[0]/4 + j * e_ad.step[1]/4);

            bool left      = (j == 0 ? 0.0f : -*(grad_x_ptr  + i * grad_x.step[0]/4  + (j-1) * grad_x.step[1]/4))  < -*(e_x_ptr  + i * e_x.step[0]/4  + j * e_x.step[1]/4);
            bool up        = (i == 0 ? 0.0f : -*(grad_y_ptr  + (i-1) * grad_y.step[0]/4  + j * grad_y.step[1]/4))  < -*(e_y_ptr  + i * e_y.step[0]/4  + j * e_y.step[1]/4);
            bool downright = (i+1 >= result.rows || j+1 >= result.cols) ? 0.0f : 
                        -*(grad_d_ptr + (i+1) * grad_d.step[0]/4 + (j+1) * grad_d.step[1]/4) < -*(e_d_ptr + i * e_d.step[0]/4 + j * e_d.step[1]/4);
            bool downleft = (i+1 >= result.rows || j == 0) ? 0.0f : 
                        -*(grad_ad_ptr + (i+1) * grad_ad.step[0]/4 + (j-1) * grad_ad.step[1]/4) < -*(e_ad_ptr + i * e_ad.step[0]/4 + j * e_ad.step[1]/4);



            unsigned char lbp = down + upleft * 2 + right * 4 + upright * 8 + up * 16 + downright * 32 + left * 64 + downleft * 128;
            *(ptr+result.step[0] * i + result.step[1] * j + l*2+1) = lbp;
            }
        }
    }
}

cv::Mat 
LTPComputer::compute(const cv::Mat &rgb_image) const
{
    cv::Mat grad_x = gx(rgb_image);
    cv::Mat grad_y = gy(rgb_image);
    cv::Mat grad_d = gd(rgb_image);
    cv::Mat grad_ad = gad(rgb_image);

    cv::Mat e_x = cv::abs(grad_x);
    cv::Mat e_y = cv::abs(grad_y);
    cv::Mat e_d = cv::abs(grad_d);
    cv::Mat e_ad = cv::abs(grad_ad);

    cv::GaussianBlur(e_x, e_x, cv::Size(17,17), 5);
    cv::GaussianBlur(e_y, e_y, cv::Size(17,17), 5);
    cv::GaussianBlur(e_d, e_d, cv::Size(17,17), 5);
    cv::GaussianBlur(e_ad, e_ad, cv::Size(17,17), 5);

    cv::Mat result(rgb_image.rows, rgb_image.cols, CV_8UC(16));

    for (int i = 0; i < _nmaps; i++) {
        float l = -log(1-float (i+1)/(float(_nmaps)+1.0f));

        make_ltp(result, i, e_x * l, e_y * l, e_d * l, e_ad * l, grad_x, grad_y, grad_d, grad_ad);
    }
    return result;
}

/*
cv::Mat
LTPComputer::get_vector(const cv::Mat &lbp_map, int x, int y, int ex, int ey) const
{
    cv::Mat patch = lbp_map.colRange(x, ex).rowRange(y, ey);
    int w = ex - x;
    int h = ey - y;
    assert(w == 24);
    assert(h == 12);
    cv::Mat vec(1, w * h * _nmaps * 2, CV_32FC1);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int c = 0; c < _nmaps * 2; c++) {
                vec.at<float>(0, c*w*h + i*w + j) = *(patch.ptr<unsigned char>(i,j) + c);
            }
        }
    }
    return vec;
}
*/

/*
int main(int argc, const char *argv[])
{
    cv::Mat img = cv::imread("../test_icdar_2005/104.jpg");
    cv::resize(img, img, cv::Size(), 0.050, 0.050);
    LTPComputer l(8);
    std::vector<cv::Mat> chans;
    cv::Mat result = l.compute(img);
    cv::split(result, chans);
    cv::imshow("C1", chans[11]); cv::waitKey(0);
    return 0;
}
*/

}

