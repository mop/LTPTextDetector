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

#ifndef IMAGEPYRAMID_H

#define IMAGEPYRAMID_H

#include <opencv2/core/core.hpp>

template <class T>
class ImagePyramid 
{
public:
    ImagePyramid(int levels, float scale_step, int window_height, const cv::Mat &image)
    : _levels(levels), _scale_step(scale_step), _window_height(window_height), _original_height(image.rows)
    {
        create_maps(image);
    }
    ~ImagePyramid() {}

    cv::Mat compute_feature(const cv::Point &start, const cv::Point &end) const
    {
        // determine scale
        int s = get_scale(end.y - start.y + 1);
        float scale_factor = pow(_scale_step, s);

        // sanity checks
        s = std::min(_levels, s);
        s = std::max(0, s);

        cv::Point scaled_start = start * scale_factor;
        cv::Point scaled_end   = end   * scale_factor;

        scaled_end.x = cv::min(scaled_end.x, _feature_maps[s].get_size().width - 1);
        scaled_end.y = cv::min(scaled_end.y, _feature_maps[s].get_size().height - 1);
        scaled_start.x = std::min(scaled_start.x, scaled_end.x);
        scaled_start.y = std::min(scaled_start.y, scaled_end.y);

        // compute the thing
        return _feature_maps[s].compute_feature(scaled_start, scaled_end);
    }


private:
    int get_scale(int height) const
    {
        // we need to rescale original the window to 'scale' in order to fit it perfectly to the box
        float scale = float (_window_height) / float (height);

        // determine the nearest integer which corresponds to this scale
        float scale_factor = 1;
        for (int i = 0; i < _levels; i++) {
            scale_factor = scale_factor * _scale_step;
            if (scale_factor < scale) return i;
            if (i + 1 < _levels && _feature_maps[i+1].get_size().width <= 0) {
                return i; // next level not existing
            }
        }
        return _levels - 1;
    }

    void create_maps(const cv::Mat &image)
    {
        _feature_maps.resize(_levels);
        #pragma omp parallel for
        for (int i = 0; i < _levels; i++) {
            float scale = 1.0 * pow(_scale_step, i);

            if (image.rows * scale < 2*_window_height || 
                image.cols * scale < 2 * _window_height)  { 
                continue;
            }

            cv::Mat rescaled_img = image;
            if (i > 0) {
                cv::resize(image, rescaled_img, cv::Size(), scale, scale);
            } else {
                rescaled_img = image;
            }

            T map;
            map.set_image(rescaled_img);
            _feature_maps[i] = map;
        }
    }

    int _levels;
    int _window_height;
    int _original_height;
    float _scale_step;
    std::vector<T> _feature_maps;
};

#endif /* end of include guard: IMAGEPYRAMID_H */
