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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <detector/Adaboost.h>
#include <text_detector/LTPComputer.h>

#include <dirent.h>
#include <getopt.h>
#include <list>

#include <fstream>
#include <memory>

#include <text_detector/utils.h>
#include <text_detector/config.h>

#include <boost/filesystem.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/timer/timer.hpp>

namespace fs = boost::filesystem;

static cv::Mat 
classify_single_scale(
        const cv::Mat &image,
        const std::shared_ptr<Detector::Adaboost> &trees,
        const cv::Size &window_size,
        const cv::Size &shift_size,
        bool sample_false_positives,
        std::list<cv::Mat> &false_positives,
        float thresh,
        int scale
)
{
    double win_width = window_size.width;
    double win_height = window_size.height;
    double shift_width = shift_size.width;
    double shift_height = shift_size.height;

    double img_width = image.cols;
    double img_height = image.rows;

    int n_width_shifts = ceil((img_width - win_width) / shift_width) + 1;
    int n_height_shifts = ceil((img_height - win_height) / shift_height) + 1;

    cv::Mat mags, qangle;
    TextDetector::LTPComputer ltp;
    cv::Mat ltp_maps = ltp.compute(image);
    ltp_maps.convertTo(ltp_maps, CV_64FC(16));
    std::vector<cv::Mat> feature_maps;
    cv::split(ltp_maps, feature_maps);

    cv::Mat result_image(image.rows, image.cols, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat norm_image(image.rows, image.cols, CV_32FC1, cv::Scalar(0.0));

    // go through all shifts
    #pragma omp parallel for
    for (int i = 0; i < n_height_shifts; ++i) {
        int y = i*shift_height;

        for (int j = 0; j < n_width_shifts; ++j) {
            int x = j * shift_width;
            int actual_x = x;
            int actual_y = y;
            if ((x + win_width) >= img_width) {
                actual_x = img_width - win_width;
            }
            if ((y + win_height) >= img_height) {
                actual_y = img_height - win_height;
            }

            //float result = 2.0 * trees.predict_prob(f) - 1.0;
            float result = 1.0/(1.0+exp(-trees->predict(feature_maps, actual_x, actual_y, Detector::Adaboost::LAZY) + 0)) * 2.0 - 1.0;

            #pragma omp critical
            {
            norm_image.rowRange(actual_y, actual_y + win_height).colRange(actual_x, actual_x + win_width) += cv::Scalar(1.0);
            if (result > 0) {
                result_image.rowRange(actual_y, actual_y + win_height).colRange(actual_x, actual_x + win_width) += cv::Scalar(result);
                
                if (sample_false_positives && result > thresh) {
                    cv::Mat f = ltp.get_vector<double>(ltp_maps, actual_x, actual_y, actual_x + win_width, actual_y + win_height);
                    cv::Mat fps_vec(1, f.cols + 5, CV_32FC1, cv::Scalar(0.0f));
                    cv::Mat sub = fps_vec.colRange(5,fps_vec.cols);
                    f.copyTo(sub);

                    fps_vec.at<float>(0, 1) = result;
                    fps_vec.at<float>(0, 2) = scale;
                    fps_vec.at<float>(0, 3) = actual_x;
                    fps_vec.at<float>(0, 4) = actual_y;
                    false_positives.push_back(fps_vec);
                }
            }
            }
        }
    }

    norm_image += cv::Scalar(1e-10);
    for (int i = 0; i < result_image.rows; i++) {
        for (int j = 0; j < result_image.cols; j++) {
            result_image.at<float>(i,j) /= norm_image.at<float>(i,j);
        }
    }

    return result_image;
}

static void 
bootstrap_image(
        const std::string &image_path, 
        const std::string &result_path, 
        const std::shared_ptr<Detector::Adaboost> &trees,
        std::list<cv::Mat> &false_positives, 
        float thresh, 
        const cv::Size &window_size, 
        const cv::Size &shift_size,
        float scale_ratio, 
        int num_scales,
        bool sample_false_positives)
{
    cv::Mat image = cv::imread(image_path);
    std::vector<std::string> parts = split(image_path, '/');
    std::string filename = parts[parts.size()-1];
    std::string name = filename.substr(0, filename.size()-4);
    
    // classify each scale...
    cv::Size original_size(image.cols, image.rows);
    std::vector<cv::Mat> results;
    for (int i = 0; i < num_scales; ++i) {

        std::cout << "scale: " << i << std::endl;
        if (image.rows <= window_size.height || image.cols <= window_size.width) break;

        cv::Mat result = classify_single_scale(
            image, 
            trees, 
            window_size, shift_size, 
            sample_false_positives,
            false_positives, 
            thresh, 
            i);
        results.push_back(result);

        cv::resize(image, image, cv::Size(), scale_ratio, scale_ratio);
    }

    std::cout << "Writing responses...." << std::endl;
    cv::Mat response(original_size.height, original_size.width, CV_32FC1, cv::Scalar(0.0f));
    for (unsigned int i = 0; i < results.size(); ++i) {
        cv::Mat tmp;
        cv::GaussianBlur(results[i], results[i], cv::Size(5,5), 2);
        cv::resize(results[i], tmp, cv::Size(original_size.width, original_size.height), 0, 0, cv::INTER_NEAREST);
        response = response + tmp;

        //double min, max;
        //cv::minMaxIdx(results[i], &min, &max);
        cv::multiply(results[i], cv::Scalar::all(255.0), results[i]);
    }
    double min, max;
    cv::minMaxIdx(response, &min, &max);
    cv::multiply(response, cv::Scalar::all(255.0/max), response);

    {
    std::stringstream ss;
    ss << result_path << "/" << name << "_response.png";
    std::cout << "writing: " << result_path << "/" << name << "_response.png" << std::endl;
    cv::imwrite(ss.str(), response);
    }
    {
    std::stringstream ss;
    ss << result_path << "/" << name << "_scale.txt";
    std::cout << "writing: " << result_path << "/" << name << "_scale.txt: " << min << " " << max << std::endl;
    std::ofstream ofs(ss.str().c_str());
    ofs << min << "," << max << std::endl;
    }

    for (unsigned int i = 0; i < results.size(); ++i) {
        std::stringstream ss;
        ss << result_path << "/" << name << "_response_all_" << i << ".png";
        std::cout << "writing: " << result_path << "/" << name << "_response_all_" << i << ".png" << std::endl;
        cv::imwrite(ss.str(), results[i]);
    }
}

static bool
is_intersection(const cv::Mat &a, const cv::Mat &b)
{
    if (a.at<float>(0, 2) != b.at<float>(0,2))
        return false;

    cv::Rect x(a.at<float>(0,3), a.at<float>(0,4), 24, 12);
    cv::Rect y(b.at<float>(0,3), b.at<float>(0,4), 24, 12);
    cv::Rect intersect = x & y;
    cv::Rect u = x | y;
    return float(intersect.width * intersect.height) / float(u.width *u.height) > 0.0f;
}
static 
std::vector<cv::Mat> filter_false_positives(const std::vector<cv::Mat> &false_positives, int max)
{
    std::vector<bool> suppressed(false_positives.size(), 0);
    std::vector<int> scale_counter(50,0);
    int max_per_scale = 50;
    std::vector<cv::Mat> result;
    result.reserve(max);

    for (size_t i = 0; i < false_positives.size() && result.size() < max; i++) {
        if (suppressed[i]) continue;
        if (scale_counter[int(false_positives[i].at<float>(0,2))] > max_per_scale) continue;
        result.push_back(false_positives[i]);
        scale_counter[int(false_positives[i].at<float>(0,2))]++;

        for (size_t j = i+1; j < false_positives.size(); j++) {
            if (is_intersection(false_positives[i], false_positives[j])) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    int c;
    std::vector<std::string> train_paths;
    std::vector<std::string> result_paths;
    std::string samples_path;
    std::string model_path;
    std::string result_path;
    int batch_size = 10000;
    int max_per_image = 100;

    int window_width = 24;
    int window_height = 12;
    int shift_width = 4;
    int shift_height = 4;


    float scale_ratio = 2/3.0f;
    int num_scales = 10;
    int upper_limit = -1;

    std::vector<int> excludes;


    while ((c = getopt(argc, argv, "u:t:g:r:m:i:b:n:d:v:f:w:s:h")) != -1) {
        switch (c) {
            case 't':
                train_paths = split(optarg,',');
                break;
            case 'r':
                result_paths = split(optarg,',');
                break;
            case 'e':
                excludes = spliti(optarg,',');
                break;
            case 'u':
                std::stringstream(optarg) >> upper_limit;
                break;
            case 'm':
                model_path = optarg;
                break;
            case 's':
                {
                std::vector<int> tmp(spliti(optarg, 'x'));
                shift_width = tmp[0];
                shift_height = tmp[1];
                }
                break;
            case 'f':
                {
                    result_path = optarg;
                }
                break;
            case 'w':
                {
                std::vector<int> tmp(spliti(optarg, 'x'));
                window_width = tmp[0];
                window_height = tmp[1];
                }
                break;
            case 'b':
                {
                    std::stringstream ss(optarg);
                    ss >> batch_size;
                }
                break;
            case 'h':
                std::cout << "Usage: bootstrap OPTIONS " << std::endl
                    << "\t -t <train paths>" << std::endl
                    << "\t -r <result paths>" << std::endl
                    << "\t -e <excludes>" << std::endl
                    << "\t -m <model file>" << std::endl
                    << "\t -f <false positive result file>" << std::endl
                    << "\t -d <maximum depth of trees>" << std::endl
                    << "\t -u <upper limit>" << std::endl
                    << "\t -s <shift_width>x<shift_height>" << std::endl
                    << "\t -w <window_width>x<window_height>" << std::endl
                    << "\t -b <batch size>" << std::endl;
            default:
                break;
        }
    }

    if (train_paths.empty()) {
        std::cerr << "Training path must not be empty" << std::endl;
        return 1;
    }
    if (model_path == "") {
        std::cerr << "Need a model file" << std::endl;
        return 1;
    }

    //if (result_path == "") {
    //    std::cerr << "Need a result path" << std::endl;
    //    return 1;
    //}

    bool sample_false_positives = result_path != "";
    if (train_paths.size() != result_paths.size()) {
        std::cout << train_paths.size() << " " << result_paths.size() << " " << std::endl;
        std::cerr << "need to have as many result paths as training paths as gt paths" << std::endl;
        return 1;
    }

    std::shared_ptr<Detector::Adaboost> rtrees(new Detector::Adaboost());

    std::ifstream ifs(model_path);
    boost::archive::text_iarchive ia(ifs);
    ia >> *rtrees;
    //rtrees->load(model_path);

    float thresh = 0.1f;
    std::list<cv::Mat> false_positives;

    std::ofstream ofs;
    if (sample_false_positives)
        ofs.open(result_path);

    for (unsigned int i = 0; i < train_paths.size(); ++i) {
        std::string train_path = train_paths[0];
        if (!fs::is_directory(train_path)) {
            std::cerr << "train path: " << train_path << " does not exist" << std::endl;
            continue;
        }

        std::vector<fs::path> files;
        std::copy(fs::directory_iterator(train_path), fs::directory_iterator(), 
            std::back_inserter(files));
        std::sort(files.begin(), files.end());

        for (fs::path file : files) {
            if (file.extension() != ".jpg" && file.extension() != ".png") {
                continue;
            }
            int idx = -1;
            std::stringstream(file.stem().generic_string()) >> idx;

            if (std::find(excludes.begin(), excludes.end(), idx) != excludes.end()) {
                continue;
            }
            if (upper_limit != -1 && idx > upper_limit) continue;
            std::string image = file.generic_string();

            std::cout << "bootstrapping: " << image << std::endl;
            boost::timer::cpu_timer t;
            bootstrap_image(
                image, result_paths[i],
                rtrees,
                false_positives,
                thresh,
                cv::Size(window_width, window_height),
                cv::Size(shift_width, shift_height),
                scale_ratio,
                num_scales,
                sample_false_positives);
            std::cout << "bootstrapped in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
            std::vector<cv::Mat> v(false_positives.begin(), false_positives.end());
            for (size_t j = 0; j < v.size(); j++) {
                int idx = std::min(int (v.size()-1), std::max(0, int(rand() % v.size())));
                std::swap(v[j], v[idx]);
            }
            v = filter_false_positives(v, max_per_image);
            std::cout << "False Positives: " <<  v.size() << std::endl;
            for (size_t j = 0; j < v.size() && j < max_per_image; j++) {
                v[j].at<float>(0,0) = idx;
                ofs << v[j].at<float>(0,0);
                for (int k = 1; k < v[j].cols; k++) {
                    ofs << "," << v[j].at<float>(0,k);
                }
                ofs << std::endl;
            }
            false_positives.clear();
            v.clear();
        }
    }
    
    ofs.close();
    return 0;
}
