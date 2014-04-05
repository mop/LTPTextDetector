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
#include <text_detector/CCUtils.h>

#include <iostream>
#include <fstream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <text_detector/config.h>

namespace TextDetector {

std::vector<cv::Point> load_pixels(const fs::path &path)
{
    if (!fs::exists(path)) {
        std::cout << "Error: " << path << " does not exist!" << std::endl;
        return std::vector<cv::Point>();
    }
    std::vector<cv::Point> result;
    std::ifstream ifs(path.generic_string());
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string part;
        std::vector<int> pt;
        while (std::getline(ss, part, ',')) {
            int coord;
            std::stringstream(part) >> coord;
            pt.push_back(coord);
        }

        result.push_back(cv::Point(pt[0], pt[1]));
    }
    return result;
}

static std::vector<int>
lookup_neighbors(
	const std::vector<std::vector<int> > &lookup_table,
	const std::vector<int> &idxs)
{
    std::vector<int> result;
    for (size_t i = 0; i < idxs.size(); i++) {
        int idx = idxs[i];
        if (!lookup_table[idx].empty()) {
            result.insert(result.end(), lookup_table[idx].begin(), lookup_table[idx].end());
        }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

static std::vector<int>
find_indices(const cv::Mat &img, int val)
{
    std::vector<int> result;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<float>(i,j) == val) {
                result.push_back(i*img.cols + j);
            }
        }
    }
    return result;
}

cv::Mat compute_swt(const cv::Mat &bw_img)
{
    cv::Mat dist;
    cv::Mat bw;
    bw_img.convertTo(bw, CV_8UC1);

    // crop the image to speedup computation!!!
    cv::Point min_loc(bw_img.cols, bw_img.rows);
    cv::Point max_loc(0,0);

    for (int i = 0; i < bw_img.rows; i++) {
        for (int j = 0; j < bw_img.cols; j++) {
            if (bw.at<unsigned char>(i,j) > 0) {
                if (min_loc.x > j) min_loc.x = j;
                if (min_loc.y > i) min_loc.y = i;
                if (max_loc.x < j) max_loc.x = j;
                if (max_loc.y < i) max_loc.y = i;
            }
        }
    }

    min_loc.y = std::max(0, min_loc.y - 5);
    min_loc.x = std::max(0, min_loc.x - 5);
    max_loc.y = std::min(bw_img.rows - 1, max_loc.y + 5);
    max_loc.x = std::min(bw_img.cols - 1, max_loc.x + 5);

    bw = bw.rowRange(min_loc.y, max_loc.y + 1).colRange(min_loc.x, max_loc.x + 1);

    cv::distanceTransform(bw, dist, CV_DIST_L2, 5);
    //cv::Mat tmp;
    //double mi, ma;
    //cv::minMaxLoc(dist, &mi, &ma);
    //dist.convertTo(tmp, CV_8UC1, 255.0/ma);
    //cv::imshow("DIST", tmp);

    int max = 0;
    for (int i = 0; i < dist.rows; i++) {
        for (int j = 0; j < dist.cols; j++) {
            dist.at<float>(i,j) = std::floor(dist.at<float>(i,j) + 0.5);
            if (dist.at<float>(i,j) > max)
                max = dist.at<float>(i,j);
        }
    }

    // foreach foreground pixel in dist
    std::vector< std::vector<int> > lookup;
    lookup.resize(dist.rows * dist.cols);

    for (int i = 0; i < dist.rows; i++) {
        for (int j = 0; j < dist.cols; j++) {
            if (dist.at<float>(i,j) == 0.0) continue;
            float val = dist.at<float>(i,j);
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    int xc = std::min(dist.cols - 1, std::max(x + j, 0));
                    int yc = std::min(dist.rows - 1, std::max(y + i, 0));
                    if (xc == j && yc == i) continue;

                    if (dist.at<float>(yc, xc) < val && dist.at<float>(yc,xc) > 0.0) {
                        // add the indices of the neighbors...
                        lookup[i*dist.cols + j].push_back(xc + yc * dist.cols);
                    }
                }
            }
        }
    }

    for (int stroke = max; stroke >= 1; stroke--) {
        std::vector<int> indices = find_indices(dist, stroke);
        std::vector<int> neighbors = lookup_neighbors(lookup, indices);
        while (!neighbors.empty()) {
            for (int i = 0; i < neighbors.size(); i++) {
                dist.at<float>(neighbors[i] / dist.cols, neighbors[i] % dist.cols) = stroke;
            }

            neighbors = lookup_neighbors(lookup, neighbors);
        }
    }

    cv::Mat result(bw_img.rows, bw_img.cols, CV_32FC1, cv::Scalar(0.0));
    cv::Mat submat = result.rowRange(min_loc.y, max_loc.y + 1).colRange(min_loc.x, max_loc.x + 1);
    dist.copyTo(submat);
    
    return result;
}


struct MyRay {
	MyRay(const std::vector<cv::Point> &pts)
	: points(pts) {}
	std::vector<cv::Point> points;
};

static void do_swt(
	const cv::Mat &edge_image,
	const cv::Mat &gx,
	const cv::Mat &gy,
	bool dark_on_light,
	cv::Mat &swt_image,
	std::vector<MyRay> &rays)
{
	// shoot rays
    for (int i = 0; i < edge_image.rows; i++ ){
    	const unsigned char *ptr = edge_image.ptr<unsigned char>(i, 0);
        for (int j = 0; j < edge_image.cols; j++){
        	if (*ptr > 0) {
                std::vector<cv::Point> points;
                points.push_back(cv::Point(j, i));

                float grad_x = gx.at<float>(i, j);
                float grad_y = gy.at<float>(i, j);
                // normalize gradient
                float mag = sqrt((grad_x * grad_x) + (grad_y * grad_y));
                if (dark_on_light){
                    grad_x = -grad_x/mag;
                    grad_y = -grad_y/mag;
                } else {
                    grad_x = grad_x/mag;
                    grad_y = grad_y/mag;
                }
                bresenham_loop2(j, i, grad_x, grad_y, 
                    [&edge_image, &swt_image, &points, &gx, &gy, 
                     &grad_x, &grad_y, &dark_on_light, &rays](int x, int y) -> bool {
                    if (x < 0 || (x >= swt_image.cols) || y < 0 || (y >= swt_image.rows)) {
                        return false;
                    }
                    points.push_back(cv::Point(x, y));

                    // found a boundary
                    if (edge_image.at<unsigned char>(y, x) > 0) {
                        float grad_x_other = gx.at<float>(y, x);
                        float grad_y_other = gy.at<float>(y, x);
                        float mag = sqrt((grad_x_other * grad_x_other) +
                            (grad_y_other * grad_y_other));
                        if (dark_on_light) {
                            grad_x_other = -grad_x_other/mag;
                            grad_y_other = -grad_y_other/mag;
                        } else {
                            grad_x_other = grad_x_other/mag;
                            grad_y_other = grad_y_other/mag;
                        }

                        const float grad_sums = grad_x * -grad_x_other +
                        	grad_y * -grad_y_other;
                        if (acos(grad_sums) < M_PI/2.0) {
                        	const float dx = points.front().x - points.back().x;
                        	const float dy = points.front().y - points.back().y;
                            const float length = sqrt(dx * dx + dy * dy);
                            for (auto pit = points.begin(); pit != points.end(); pit++) {
                            	if (swt_image.at<float>(pit->y, pit->x) < 0) {
                            		swt_image.at<float>(pit->y, pit->x) = length;
                            	} else {
                            		const float sw =
                            			swt_image.at<float>(pit->y, pit->x);
                            		swt_image.at<float>(pit->y, pit->x) = std::min(length, sw);
                            	}
                            }

                            rays.push_back(MyRay(points));
                        }
                        return false;
                    }

                    return true;

                });
            }
            ptr++;
        }
    }
}

static void swt_median_filter(cv::Mat &swt_image, const std::vector<MyRay> &rays)
{
    for (auto rit = rays.begin(); rit != rays.end(); rit++) {
    	std::vector<float> strokes(rit->points.size());
    	int i = 0;
        for (auto pit = rit->points.begin(); pit != rit->points.end(); pit++) {
        	strokes[i++] = swt_image.at<float>(pit->y, pit->x);
        }
        std::sort(strokes.begin(), strokes.end());
        const float median = (strokes[strokes.size()/2]);
        for (auto pit = rit->points.begin(); pit != rit->points.end(); pit++) {
        	swt_image.at<float>(pit->y, pit->x) = std::min(
                swt_image.at<float>(pit->y, pit->x),
                median);
        }
    }
}

void swt(const cv::Mat &input_image,
        cv::Mat &black_on_white,
		cv::Mat &white_on_black)
{
    cv::Mat blurred, g_x, g_y, g;
    cv::GaussianBlur(input_image, blurred, cv::Size(5,5), 1.2);
    cv::Sobel(blurred, g_x, CV_16SC1, 1, 0);
    cv::Sobel(blurred, g_y, CV_16SC1, 0, 1);
    cv::multiply(g_x, g_x, g_x);
    cv::multiply(g_y, g_y, g_y);
    g = g_x + g_x;
    g.convertTo(g, CV_32FC1);
    cv::sqrt(g, g);
    cv::Mat hist;
    int hist_size = 64;

    double mi, ma;
    cv::minMaxLoc(g, &mi, &ma);
    const float range[] = {  0.0f, std::max(0.0f + 1e-2f, float(ma))  };
    const float* hist_range = { range };
    cv::calcHist(&g, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range);

    // 95-percentile
    int size = input_image.rows * input_image.cols;
    float accum = 0;
    int idx = -1;
    for (int i = 0; i < hist.rows; i++) {
        accum += hist.at<float>(i,0);
        if (accum > size * 0.95) {
            idx = i+1;
            break;
        }
    }
    float thresh = float(idx) / 64 * ma;

    double threshold_low = 0.4 * thresh;
    double threshold_high = thresh;
    blurred.convertTo(blurred, CV_8UC1);

    cv::Mat edge_image;
    cv::Canny(blurred, edge_image, threshold_low, threshold_high, 3);

    cv::Mat gaussian_image;
    input_image.convertTo(gaussian_image, CV_32FC1, 1.0 / 255.0);
    cv::GaussianBlur(gaussian_image, gaussian_image, cv::Size(5, 5), 0.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(gaussian_image, g_x, CV_32FC1, 1, 0, -1, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(gaussian_image, g_y, CV_32FC1, 0, 1, -1, 1, 0, cv::BORDER_REPLICATE);
    cv::medianBlur(g_x, g_x, 3);
    cv::medianBlur(g_y, g_y, 3);

    // Calculate SWT and return ray vectors
    {
    std::vector<MyRay> rays;
    cv::Mat swt_image(input_image.rows, input_image.cols, CV_32FC1, cv::Scalar(-1.0f));
    do_swt(edge_image, g_x, g_y, true, swt_image, rays);
    swt_median_filter(swt_image, rays);

    black_on_white = swt_image;
    }

    {
    std::vector<MyRay> rays;
    cv::Mat swt_image(input_image.rows, input_image.cols, CV_32FC1, cv::Scalar(-1.0f));
    do_swt(edge_image, g_x, g_y, false, swt_image, rays);
    swt_median_filter(swt_image, rays);

    white_on_black = swt_image;
    }
}

void compute_swt(const cv::Mat &gray_img, cv::Mat &black_on_white, cv::Mat &white_on_black)
{
	swt(gray_img, black_on_white, white_on_black); return;
}

void MserElement::compute_hog_features(const cv::Mat &gray_image)
{
    float area = 0.0;
    cv::Mat bw_img = compute_binary_image(cv::Size(gray_image.cols, gray_image.rows), area);
    cv::HOGDescriptor hog(
        cv::Size(28,28),
        cv::Size(8,8),
        cv::Size(4,4),
        cv::Size(4,4), 8);
    std::vector<float> descriptors;
    std::vector<cv::Point> locations;
    cv::Mat subimg;
    cv::Mat subimg_gray;
    int extra_pad = int(ceil(3.0/22.0 * std::max(_bounding_rect.width, _bounding_rect.height)  ));
    extra_pad = 0;
    if (_bounding_rect.width < _bounding_rect.height) {
        float to_pad = _bounding_rect.height - _bounding_rect.width;
        float start_x = _bounding_rect.x - to_pad/2 - extra_pad;
        float end_x = _bounding_rect.x + _bounding_rect.width + to_pad/2 + extra_pad;
        float start_y = std::max(0, _bounding_rect.y - extra_pad);
        float end_y = std::min(gray_image.rows, _bounding_rect.y + _bounding_rect.height + extra_pad);
        if (start_x < 0) {
            end_x += (-start_x);
            start_x = 0;
        }
        if (end_x > gray_image.cols) {
            start_x = start_x - (end_x - gray_image.cols);
            end_x = gray_image.cols;
        }
        start_x = std::max(0.0f, start_x);
        subimg = bw_img.rowRange(start_y, end_y).
            colRange(int (start_x), int (end_x));
        subimg_gray = gray_image.rowRange(start_y, end_y).
            colRange(int (start_x), int (end_x));
    } else {
        float to_pad = _bounding_rect.width - _bounding_rect.height;
        float start_y = _bounding_rect.y - to_pad/2 - extra_pad;
        float end_y = _bounding_rect.y + _bounding_rect.height + to_pad/2 + extra_pad;
        float start_x = std::max(0, _bounding_rect.x - extra_pad);
        float end_x = std::min(gray_image.cols, _bounding_rect.x + _bounding_rect.width + extra_pad);
        if (start_y < 0) {
            end_y += (-start_y);
            start_y = 0;
        }
        if (end_y > gray_image.rows) {
            start_y = start_y - (end_y - gray_image.rows);
            end_y = gray_image.rows;
        }
        start_y = std::max(0.0f, start_y);
        subimg = bw_img.rowRange(int (start_y), int (end_y)).
            colRange(start_x, end_x);
        subimg_gray = gray_image.rowRange(int (start_y), int (end_y)).
            colRange(start_x, end_x);
    }
    cv::resize(subimg, subimg, cv::Size(28,28));
    cv::resize(subimg_gray, subimg_gray, cv::Size(28,28));

    _binary_image = cv::Mat(1, 28*28, CV_32FC1);
    if (!subimg.isContinuous()) {
        subimg = subimg.clone();
    }
    subimg.reshape(1,28*28).copyTo(_binary_image);
    cv::transpose(_binary_image, _binary_image);

    cv::Mat subimg_flt;
    subimg.convertTo(subimg_flt, CV_32FC1, 1/255.0f);
    subimg_flt = subimg_flt > 0.0f;
    //hog.compute(subimg, descriptors, cv::Size(0,0), cv::Size(0,0), locations);

    //cv::imshow("BIN", _binary_image.reshape(0, 28));
    //cv::waitKey(0);

    //_hog_features = cv::Mat(1,descriptors.size(), CV_32FC1);
    //for (size_t i = 0; i < descriptors.size(); i++) {
    //    _hog_features.at<float>(0,i) = descriptors[i];
    //}
}

void MserElement::compute_features(
        const cv::Mat &original_image,
        const cv::Mat &gradient_image,
        const cv::Mat &swt1, const cv::Mat &swt2) 
{
    assert(gradient_image.type() == CV_32FC1);
    assert(gradient_image.rows == original_image.rows && 
           gradient_image.cols == original_image.cols);
    cv::Rect rect(_bounding_rect);
    // aspect ratio
    _aspect = float(rect.width) / (rect.height);

    // area
    float area = 0.0;
    cv::Mat bw_img = compute_binary_image(cv::Size(original_image.cols, original_image.rows), area);

    cv::Mat mask1, mask2;
    cv::Mat swt1_roi = swt1.rowRange(
        _bounding_rect.y, _bounding_rect.y + _bounding_rect.height
    ).colRange(_bounding_rect.x, _bounding_rect.x + _bounding_rect.width);
    cv::Mat swt2_roi = swt2.rowRange(
        _bounding_rect.y, _bounding_rect.y + _bounding_rect.height
    ).colRange(_bounding_rect.x, _bounding_rect.x + _bounding_rect.width);
    cv::Mat bw_roi = bw_img.rowRange(
        _bounding_rect.y, _bounding_rect.y + _bounding_rect.height
    ).colRange(_bounding_rect.x, _bounding_rect.x + _bounding_rect.width);

    mask1 = swt1_roi >= 0;
    mask2 = swt2_roi >= 0;

    mask1 = mask1 & bw_roi;
    mask2 = mask2 & bw_roi;

    cv::Scalar swt1_mean, swt1_stddev, swt2_mean, swt2_stddev;
    cv::meanStdDev(swt1_roi, swt1_mean, swt1_stddev, mask1);
    cv::meanStdDev(swt2_roi, swt2_mean, swt2_stddev, mask2);

    float swt_stddev;
    if (cv::sum(mask1)[0] > cv::sum(mask2)[0]) {
        swt_stddev = swt1_stddev[0];
        _swt_mean = swt1_mean[0];
    } else {
        swt_stddev = swt2_stddev[0];
        _swt_mean = swt2_mean[0];
    }

    // horizontal crossings
    int y_top = rect.y + rect.height * 0.2;
    int y_middle = rect.y + rect.height * 0.5;
    int y_bottom = rect.y + rect.height * 0.8;

    _crossings_top = _crossings_middle = _crossings_bottom = 0.0f;
    for (int x = rect.x; x < rect.x + rect.width - 2; ++x) {
        if (bw_img.at<unsigned char>(y_top, x) != bw_img.at<unsigned char>(y_top, x + 1)) {
            _crossings_top += 1.0;
        }
        if (bw_img.at<unsigned char>(y_middle, x) != bw_img.at<unsigned char>(y_middle, x + 1)) {
            _crossings_middle += 1.0;
        }
        if (bw_img.at<unsigned char>(y_bottom, x) != bw_img.at<unsigned char>(y_bottom, x + 1)) {
            _crossings_bottom += 1.0;
        }
    }
    
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    int pad = 8;
    cv::Mat pad_bw_roi;
    cv::copyMakeBorder(bw_roi, pad_bw_roi, pad/2, pad/2, pad/2, pad/2, 
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::findContours(pad_bw_roi, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    //cv::findContours(bw_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    float perimeter_size = 0.0f;
    float grad = 0.0f;
    _euler = 0;
    float hull_area = 0.0;
    int max_c = -1;
    int max_p = 0;
    float hull_perimeter = 0.0;
    float hole_area = 0.0;
    for (size_t i = 0; i < hierarchy.size(); i++) {
        if (hierarchy[i][3] < 0) { // is root level contour?!
            perimeter_size += contours[i].size();
            if (contours[i].size() >= max_p) {
                max_p = contours[i].size();
                max_c = i;
            }
            for (size_t j = 0; j < contours[i].size(); ++j) {
                grad += gradient_image.at<float>(
                    contours[i][j].y + rect.y-pad/2, 
                    contours[i][j].x + rect.x-pad/2);
                //grad += gradient_image.at<float>(
                //    contours[i][j].y, contours[i][j].x);
            }

            std::vector<cv::Point> hull;
            cv::convexHull(contours[i], hull);
            cv::Mat hull_img(rect.height+pad, rect.width+pad, CV_8UC1, cv::Scalar(0));
            //cv::Mat hull_img(bw_img.rows, bw_img.cols, CV_8UC1, cv::Scalar(0));

            std::vector<std::vector<cv::Point> > a;
            a.push_back(hull);

            cv::drawContours(hull_img, a, 0, cv::Scalar(255), CV_FILLED);
            hull_area += cv::sum(hull_img)[0] / 255.0; // more accurate

            std::vector<std::vector<cv::Point> > hull_contours;
            cv::findContours(hull_img, hull_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            for (std::vector<cv::Point> hull_contour : hull_contours) {
                hull_perimeter += hull.size();
            }

        } else {
            // count as hole if big enough
            float tmp = cv::contourArea(contours[i]);
            if (tmp > MIN_HOLE_SIZE) {
                _euler += 1.0;
                hole_area += tmp;
            }
        }
    }
    _ellipse = cv::minAreaRect(_pixels);

    float w = std::max(_ellipse.size.width, 1.0f);
    float h = std::max(_ellipse.size.height, 1.0f);

    // area ratio
    _area_ratio = float(area) / float(_bounding_rect.width * _bounding_rect.height);
    _hull_ratio = area / (hull_area+1e-5);

    _ellipse_area_ratio = float(area) / (w * h);
    _ellipse_ratio = w / h;

    if (perimeter_size > 0) {
        _bb_compactness = float(_bounding_rect.width * _bounding_rect.height) / (perimeter_size * perimeter_size);
        _compactness = float(area) / (perimeter_size * perimeter_size);
        _ellipse_compactness = float(w * h) / (perimeter_size * perimeter_size);
        _gradient = grad / perimeter_size;
    } else {
        _compactness = 0.0f;
        _ellipse_compactness = 0.0f;
        _gradient = 0.0f;
        _bb_compactness = 0.0f;
    }
    _swt_stddev = swt_stddev;

    _hull_perimeter_ratio = hull_perimeter / float (perimeter_size);
    _hole_area_ratio = hole_area / area;
}

void MserElement::compute_bounding_rect() 
{
    if (_pixels.empty()) 
        return;
    int min_x = _pixels[0].x;
    int max_x = _pixels[0].x;
    int min_y = _pixels[0].y;
    int max_y = _pixels[0].y;
    for (size_t i = 1; i < _pixels.size(); i++) {
        if (_pixels[i].x < min_x) {
            min_x = _pixels[i].x;
        }
        if (_pixels[i].y < min_y) {
            min_y = _pixels[i].y;
        }
        if (_pixels[i].x > max_x) {
            max_x = _pixels[i].x;
        }
        if (_pixels[i].y > max_y) {
            max_y = _pixels[i].y;
        }
    }

    _bounding_rect = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
}

cv::Vec3f MserElement::get_mean_color(const cv::Mat &img) const
{
    cv::Vec3f color; 
    color[0] = color[1] = color[2] = 0.0f;
    for (size_t i = 0; i < _pixels.size(); i++) {
        cv::Vec3b c = img.at<cv::Vec3b>(_pixels[i].y, _pixels[i].x);
        color[0] += float (c[0]);
        color[1] += float (c[1]);
        color[2] += float (c[2]);
    }
    color[0] /= _pixels.size();
    color[1] /= _pixels.size();
    color[2] /= _pixels.size();
    return color;
}

cv::Mat MserElement::compute_binary_image(const cv::Size &size, float &area) const
{
    area = 0.0f;
    cv::Mat bw_img(size.height, size.width, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < _pixels.size(); i++) {
        if (bw_img.at<unsigned char>(_pixels[i].y, _pixels[i].x) == 0) {
            area += 1.0f;
            bw_img.at<unsigned char>(_pixels[i].y, _pixels[i].x) = 255;
        }
    }
    return bw_img;
}

cv::Mat MserElement::compute_pairwise_features(const cv::Mat &original_image, const cv::Mat &gradient_image, const MserElement &other) const
{
    // distance (normalized)!
    float hor_dist = std::abs(_centroid[0] - other._centroid[0]) / float (std::max(_bounding_rect.width, other._bounding_rect.width));
    float ver_dist = std::abs(_centroid[1] - other._centroid[1]) / float (std::max(_bounding_rect.height, other._bounding_rect.height));
    // top & bottom difference
    float diff_top = std::abs(_bounding_rect.y - other._bounding_rect.y) / float(std::max(_bounding_rect.height, other._bounding_rect.height));
    float diff_bottom = std::abs((_bounding_rect.y + _bounding_rect.height) - (other._bounding_rect.y + other._bounding_rect.height)) / 
                        float(std::max(_bounding_rect.height, other._bounding_rect.height));
    // color difference
    cv::Vec3f col = get_mean_color(original_image);
    cv::Vec3f col_other = other.get_mean_color(original_image);
    cv::Vec3f diff = col - col_other;
    float color_difference = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / 255.0f;

    // horizontal ratio
    float hor_ratio = float(std::min(_bounding_rect.height, other._bounding_rect.height)) /
                      float(std::max(_bounding_rect.height, other._bounding_rect.height));
    // vertical ratio
    float ver_ratio = float(std::min(_bounding_rect.width, other._bounding_rect.width)) /
                      float(std::max(_bounding_rect.width, other._bounding_rect.width));

    float swt_mean_ratio = std::min(_swt_mean, other._swt_mean) / 
            std::max(1e-8f, std::max(_swt_mean, other._swt_mean));
    
    float top1 = _bounding_rect.tl().y;
    float bottom1 = _bounding_rect.br().y;
    float top2 = other._bounding_rect.tl().y;
    float bottom2 = other._bounding_rect.br().y;

    float ver_overlap = std::min(bottom1, bottom2) - std::max(top1, top2);
    float maxh = std::max(_bounding_rect.height, other._bounding_rect.height);
    ver_overlap /= maxh;

    return cv::Mat_<float>(1,9) << hor_dist, ver_dist, diff_top, diff_bottom, color_difference, hor_ratio, ver_ratio, swt_mean_ratio, ver_overlap;
}

void MserElement::compute_centroid()
{
    cv::Vec2f c;
    c[0] = 0.0f; c[1] = 0.0f;
    for (size_t i = 0; i < _pixels.size(); i++) {
        c[0] += float (_pixels[i].x) / float (_pixels.size());
        c[1] += float (_pixels[i].y) / float (_pixels.size());
    }
    
    _centroid = c;
}

cv::Mat compute_gradient_single_chan(const cv::Mat &img)
{
    cv::Mat tmp = img;
    //img.convertTo(tmp, CV_32FC1, 1/255.0);
    //cv::GaussianBlur(tmp, tmp, cv::Size(3,3), 1.5);

    
    cv::Mat result(img.rows, img.cols, CV_32FC1, cv::Scalar(0.0f));
    std::vector<int> buf(img.cols+2 + img.rows + 2, 0);
    int *xmap = &buf[1];
    int *ymap = &buf[img.cols + 2 + 1];
    for (int i = -1; i < img.cols+1; i++) {
        xmap[i] = cv::borderInterpolate(i, result.cols, cv::BORDER_REFLECT_101);
    }
    for (int i = -1; i < img.rows+1; i++) {
        ymap[i] = cv::borderInterpolate(i, result.rows, cv::BORDER_REFLECT_101);
    }

    if (img.channels() == 3) {
        xmap -= 1;
        // stride of 3
        // img.cols + 2 == width of image + left and right pixel
        for (int i = 0; i < img.cols + 2; i++) {
            xmap[i] *= 3;
        }
        xmap += 1;
    }


    std::vector<float> dbuf(img.cols * 2); // dx, dy
    for (int i = 0; i < img.rows; i++) {
        const uchar* img_ptr  = img.data + img.step*ymap[i];
        const uchar* prev_ptr = img.data + img.step*ymap[i-1];
        const uchar* next_ptr = img.data + img.step*ymap[i+1];
        float *grad_ptr = result.ptr<float>(i);
        if (img.channels() == 1) {
            for (int j = 0; j < img.cols; j++) {
                int x1 = xmap[j];
                // dx
                dbuf[j] = static_cast<float>(img_ptr[xmap[j+1]]) - 
                          static_cast<float>(img_ptr[xmap[j-1]]);
                // dy
                dbuf[j+img.cols] =
                    static_cast<float>(next_ptr[x1]) - 
                    static_cast<float>(prev_ptr[x1]);
            }
        } else {
            // stride of 3
            for (int j = 0; j < img.cols; j++) {
                int x1 = xmap[j];
                const uchar *p2 = img_ptr + xmap[j+1];
                const uchar *p0 = img_ptr + xmap[j-1];

                float dx0, dy0, dx, dy, mag0, mag;

                dx0 = p2[2] - p0[2];
                dy0 = next_ptr[x1+2] - prev_ptr[x1+2];
                mag0 = dx0*dx0 + dy0*dy0;

                dx = p2[1] - p0[1];
                dy = next_ptr[x1+1] - prev_ptr[x1+1];
                mag = dx * dx + dy * dy;
                if (mag0 < mag) {
                    mag0 = mag;
                    dx0 = dx;
                    dy0 = dy;
                }

                dx = p2[0] - p0[0];
                dy = next_ptr[x1+0] - prev_ptr[x1+0];
                mag = dx * dx + dy * dy;
                if (mag0 < mag) {
                    mag0 = mag;
                    dx0 = dx;
                    dy0 = dy;
                }

                dbuf[j] = dx0;
                dbuf[j + img.cols] = dy0;
            }
        }

        for (int j = 0; j < img.cols; j++) {
            *grad_ptr = std::sqrt(dbuf[j]*dbuf[j] + dbuf[j+img.cols] * dbuf[j+img.cols]);
            ++grad_ptr;
        }
    }

    return result;
}

cv::Mat compute_gradient(const cv::Mat &img)
{
    return compute_gradient_single_chan(img);
}

void MserElement::draw(cv::Mat &img)
{
    for (size_t i = 0; i < _pixels.size(); i++) {
        img.at<cv::Vec3b>(
            _pixels[i].y,
            _pixels[i].x
        ) = _label > 0 ? cv::Vec3b(255,0,0) : cv::Vec3b(0, 255, 0);
    }
}

void MserElement::draw(cv::Mat &img, const cv::Vec3b &color)
{
    for (size_t i = 0; i < _pixels.size(); i++) {
        img.at<cv::Vec3b>(_pixels[i].y, _pixels[i].x) = color;
    }
}

cv::Mat MserElement::get_unary_features() const 
{
    cv::Mat result(1, N_UNARY_FEATURES, CV_32FC1);
    int i = 0;
    result.at<float>(0,i++)  = get_aspect_ratio();
    result.at<float>(0,i++)  = get_gradient();
    result.at<float>(0,i++)  = get_area_ratio();
    result.at<float>(0,i++)  = get_hull_ratio();
    result.at<float>(0,i++)  = get_compactness();
    result.at<float>(0,i++)  = get_euler();
    result.at<float>(0,i++)  = get_crossings_top();
    result.at<float>(0,i++)  = get_crossings_middle();
    result.at<float>(0,i++)  = get_crossings_bottom();
    result.at<float>(0,i++)  = get_swt_stddev();
    result.at<float>(0,i++)  = get_swt_mean();

    result.at<float>(0,i++)  = get_ellipse_area_ratio();
    result.at<float>(0,i++)  = get_ellipse_compactness();
    result.at<float>(0,i++)  = get_ellipse_bb_ratio();
    result.at<float>(0,i++)  = get_bb_compactness();

    return result;
}

}
