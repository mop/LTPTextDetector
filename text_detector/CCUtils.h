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
#ifndef CCUTILS_H
#define CCUTILS_H

#define MIN_HOLE_SIZE 10.0
#define MIN_HOLE_RATIO 0.05

#include <vector>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "HogIntegralImageComputer.h"
#include "ImagePyramid.h"

#define N_HOG_DIM (1152)

namespace fs = boost::filesystem;

namespace TextDetector {

//! This class represents a (mser) blob. It computes shape features
//! used for classification.
class MserElement 
{
public:
    MserElement(int imgid = -1, int uid = -1, int label = -1,
    		const std::vector<cv::Point> &pix = std::vector<cv::Point> ())
    : _imgid(imgid), _uid(uid), _label(label), _pixels(pix)
    { compute_bounding_rect(); compute_centroid(); }
    ~MserElement() {}

    std::vector<cv::Point> get_pixels() const { return _pixels; }
    void set_label(int l) { _label = l; }

    int get_imgid() const { return _imgid; }
    int get_uid() const { return _uid; }
    int get_label() const { return _label; }
    cv::Rect get_bounding_rect() const { return _bounding_rect; }
    cv::Vec2f get_centroid() const { return _centroid; }

    void draw(cv::Mat &img);
    void draw(cv::Mat &img, const cv::Vec3b &color);

    void compute_features(const cv::Mat &original_image, const cv::Mat &gradient_image, /*const ImagePyramid<HogIntegralImageComputer> &hog_computer,*/ const cv::Mat &swt1, const cv::Mat &swt2);
    void compute_hog_features(const cv::Mat &gray_image);

    cv::Mat compute_pairwise_features(
        const cv::Mat &original_image,
        const cv::Mat &gradient_image,
        const MserElement &other) const;
    cv::Vec3f get_mean_color(const cv::Mat &img) const;

    cv::Mat get_unary_features() const;

    float get_aspect_ratio() const { return _aspect; }
    float get_gradient() const { return _gradient; }
    float get_area_ratio() const { return _area_ratio; }
    float get_hull_ratio() const { return _hull_ratio; }
    float get_compactness() const { return _compactness; }
    float get_bb_compactness() const { return _bb_compactness; }
    float get_euler() const { return _euler; }
    float get_crossings_top() const { return _crossings_top; }
    float get_crossings_middle() const { return _crossings_middle; }
    float get_crossings_bottom() const { return _crossings_bottom; }
    float get_swt_stddev() const { return _swt_stddev; }
    float get_swt_mean() const { return _swt_mean / std::min(_bounding_rect.width, _bounding_rect.height); }
    float get_raw_swt_mean() const { return _swt_mean; }

    float get_hull_perimeter_ratio() const { return _hull_perimeter_ratio; }
    float get_ellipse_area_ratio() const { return _ellipse_area_ratio; }
    float get_ellipse_compactness() const { return _ellipse_compactness; }
    float get_ellipse_bb_ratio() const { return _ellipse_ratio; }
    float get_hole_area_ratio() const { return _hole_area_ratio; }
    //cv::Mat get_hog_contour() const;
    cv::Mat get_hog_features() const { return _hog_features; }
    cv::Mat get_binary_image() const { return _binary_image; }
    void set_raw_swt_mean(float s) { _swt_mean = s; }
private:
    void compute_bounding_rect();
    void compute_centroid();
    cv::Mat compute_binary_image(const cv::Size &size, float &area) const;

    std::vector<cv::Point> _pixels;
    int _imgid;
    int _uid;
    int _label;
    cv::Rect _bounding_rect;
    cv::RotatedRect _ellipse;
    cv::Vec2f _centroid;
    //cv::Mat _swt_image;

    float _aspect;
    float _area_ratio;
    float _gradient;
    float _compactness;
    float _ellipse_compactness;
    float _bb_compactness;
    float _euler;
    float _hull_ratio;
    float _crossings_top;
    float _crossings_middle;
    float _crossings_bottom;
    float _swt_stddev;
    float _swt_mean;

    float _hull_perimeter_ratio;
    float _ellipse_area_ratio;
    float _ellipse_ratio;
    float _hole_area_ratio;
    //cv::Vec<float,9> _hog_vec;
    //cv::Mat _moments;

    cv::Mat _hog_features;
    cv::Mat _binary_image;
};

std::vector<cv::Point> load_pixels(const fs::path &path);
cv::Mat compute_gradient(const cv::Mat &img);
cv::Mat compute_gradient_single_chan(const cv::Mat &img);
cv::Mat compute_swt(const cv::Mat &img);

void compute_swt(const cv::Mat &gray_img, cv::Mat &swt1, cv::Mat &swt2);


static inline float sgn(float x) { return x < 0 ? -1 : 1; }

/**
 * This algorithm is an implementation of bresenhams line search algorithm.
 */
template <class T>
void bresenham_loop(int x, int y, float dx, float dy, const T &f)
{
    int incx = sgn(dx);
    int incy = sgn(dy);
    dx = std::abs(dx);
    dy = std::abs(dy);

    int fast_dx, fast_dy;
    float slow_dx, slow_dy;
    float delta_error;
    float error_long;
    if (dx > dy) { // move in x-direction
        fast_dx = incx; fast_dy = 0;
        slow_dx = 0; slow_dy = incy;
        delta_error = dy;
        error_long = dx;
    } else {	   // move in y-direction
        fast_dx = 0; fast_dy = incy;
        slow_dx = incx; slow_dy = 0;
        delta_error = dx;
        error_long = dy;
    }
    double error = error_long / 2.0;

    do {
    	x += fast_dx;
    	y += fast_dy;

    	error -= delta_error;
    	if (error < 0) {
    		error += error_long;
    		x += slow_dx;
    		y += slow_dy;
    	}
    } while (f(x, y));
}

template <class T>
void bresenham_loop2(int x, int y, float dx, float dy, const T &f)
{
	float step = 0.05f;
	float xf = x + 0.5f;
	float yf = y + 0.5f;
	bool result = true;
    do {
    	xf += dx * step;
    	yf += dy * step;
    	if (int(floor(xf)) != x || int(floor(yf)) != y) {
    		x = int(floor(xf)); y = int(floor(yf));
    		result = f(x,y);
    	}
    } while (result);
}

}


#endif /* end of include guard: CCUTILS_H */
