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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <math.h>
#include <getopt.h>

#include <text_detector/utils.h>


static int nheights = 3;
static int heights[] = {12,15,18};

static int window_height = 12;
static int window_width = 24;

static int window_shift_width = 12;
static int window_shift_height = 2;

static int nscales = 10;
static float scale_ratio = 2.0f/3.0f;

static int rand_patches_per_scale = 20;
static int max_rand_skips = 20;

int nexcludes = 8;
static int exclude_nums[] = {100,101,102,103,104,135,155,300};

static std::vector<cv::Rect> read_boxes(const std::string &box_path) 
{
    std::vector<cv::Rect> result;
    std::ifstream ifs(box_path.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
        std::vector<float> box = splitf(line, ',');
        if (box.size() == 4) {
            cv::Rect rect(box[0], box[1], box[2] + 1, box[3] + 1); // make the height a bit bigger
            result.push_back(rect);
        } else {
            std::cerr << "Box has not length 4!" << std::endl;
        }
    }
    return result;
}

void extract_positive_samples(const cv::Mat &img, const cv::Rect &r, std::ofstream &ofs, std::ofstream &ofs_patch)
{
    for (int i = 0; i < nheights; ++i) {
        float scale = float(heights[i]) / float(r.height);
        cv::Mat resized_img;
        cv::Mat box(1,4,CV_32FC1);
        box.at<float>(0,0) = std::max(0, r.x);
        box.at<float>(0,1) = std::max(0, r.y);
        box.at<float>(0,2) = r.width;
        box.at<float>(0,3) = r.height;

        cv::resize(img, resized_img, cv::Size(), scale, scale);
        box = box * scale;
        box.at<float>(0,2) = box.at<float>(0,2) + box.at<float>(0,0) - 1;
        box.at<float>(0,3) = box.at<float>(0,3) + box.at<float>(0,1) - 1;
        for (int j = 0; j < 4; ++j) { box.at<float>(0,j) = round(box.at<float>(0,j)); }

        box.at<float>(0,3) = std::max(box.at<float>(0,3), box.at<float>(0,1) + window_height - 1);
        box.at<float>(0,2) = std::min(box.at<float>(0,2), (float) resized_img.cols - 1); // fixup width if it is too big
        if (box.at<float>(0,3) >= resized_img.rows) {
            box.at<float>(0,3) = resized_img.rows - 1;
            box.at<float>(0,1) = box.at<float>(0,3) - window_height + 1;
        }

        if (box.at<float>(0,1) < 0) continue; 

        if ((box.at<float>(0,2) - box.at<float>(0,0) + 1) < window_width) {
            // start at the center and subtract the height of the window in the
            // positive and negative direction to avoid too small window sizes
            float center_x = box.at<float>(0,0) + round((box.at<float>(0,2) - box.at<float>(0,0) + 1.0) / 2.0);
            float x_start = int(center_x - window_width/2);
            float x_end = int(x_start + window_width - 1);

            if (x_start < 0) {
                x_start = 0;
                x_end = x_start + window_width - 1;
            }
            if (x_end >= resized_img.cols) {
                x_end = resized_img.cols - 1;
                x_start = x_end - window_width + 1;
            }
            
            if (x_start < 0) continue;

            box.at<float>(0,0) = x_start;
            box.at<float>(0,2) = x_end;
        }

        // compute hog image
        TextDetector::LTPComputer ltp;
        cv::Mat ltp_map = ltp.compute(resized_img);
        // shift the window

        int num_h_shifts = std::ceil((box.at<float>(0,2) - box.at<float>(0,0)+1) / window_shift_width);
        int num_v_shifts = std::max(1.0f, (float) std::ceil((box.at<float>(0,3) - box.at<float>(0,1) + 1.0 - window_height) / window_shift_height) + 1.0f);

        for (int h = 0; h < num_h_shifts; ++h) {
            for (int v = 0; v < num_v_shifts; ++v) {
                int x_start = box.at<float>(0,0) + window_shift_width * h;
                int x_end = x_start + window_width - 1;
                int y_start = box.at<float>(0,1) + window_shift_height * v;
                int y_end = y_start + window_height - 1;

                if (x_end > box.at<float>(0,2)) {
                    x_end = box.at<float>(0,2);
                    x_start = x_end - window_width + 1;
                }
                if (y_end > box.at<float>(0,3)) {
                    y_end = box.at<float>(0,3);
                    y_start = y_end - window_height + 1;
                }

                // extract the vector
                cv::Mat vec = ltp.get_vector<unsigned char>(ltp_map, x_start, y_start, x_end+1, y_end+1);
                assert(vec.cols == window_height*window_width*16);
                assert(vec.rows == 1);
                ofs << "1";
                for (unsigned int l = 0; l < vec.cols; ++l) {
                    ofs << "," << vec.at<float>(0,l);
                }
                ofs << std::endl;

                if (ofs_patch.is_open()) {
                    ofs_patch << "1";
                    for (unsigned int l = 0; l < 24*12; l++) {
                        int x_coord = l % 24;
                        int y_coord = l / 24;

                        cv::Vec3b col = resized_img.at<cv::Vec3b>(y_start+y_coord, x_start+x_coord);
                        ofs_patch << "," << int(col[0]) << "," << int(col[1]) << "," << int(col[2]);
                    }
                    ofs_patch << std::endl;
                }
                //std::cout << scale << " " << num_h_shifts << " " << num_v_shifts << std::endl;
                //cv::imshow("patch", resized_img.rowRange(y_start, y_end+1).colRange(x_start, x_end+1));
                //cv::waitKey(0);
            }
        }
    }
}

static std::vector<cv::Rect> 
scale_rects(const std::vector<cv::Rect> &rects, float scale)
{
    std::vector<cv::Rect> result;
    result.reserve(rects.size());
    for (unsigned int i = 0; i < rects.size(); i++) {
        cv::Rect r = rects[i];
        r.x = int(std::floor(r.x * scale));
        r.y = int(std::floor(r.y * scale));
        r.width = int(std::ceil(r.width * scale));
        r.height = int(std::ceil(r.height * scale));
        result.push_back(r);
    }
    return result;
}

static bool is_intersecting(const std::vector<cv::Rect> &rects, const cv::Rect &r)
{
    for (unsigned int i = 0; i < rects.size(); ++i) {
        cv::Rect rect = rects[i] & r;
        if (rect.height > 0 && rect.width > 0) return true;
    }
    return false;
}

static void 
extract_negative_samples(const cv::Mat &img, const std::vector<cv::Rect> &rects, float scale, std::ofstream &ofs, std::ofstream &ofs_patch)
{
    // compute hog image
    TextDetector::LTPComputer ltp;
    cv::Mat ltp_map = ltp.compute(img);

    for (unsigned int i = 0; i < rand_patches_per_scale; ++i) {
        int x_max = img.cols - window_width;
        int y_max = img.rows - window_height;
        if (x_max <= 0 || y_max <= 0) break;
        for (unsigned int j = 0; j < max_rand_skips; ++j) {
            int x = rand() % x_max;
            int y = rand() % y_max;

            cv::Rect r(x,y,window_width, window_height);
            if (!is_intersecting(rects, r)) {
                cv::Mat vec = ltp.get_vector<unsigned char>(ltp_map, r.x, r.y, r.x + r.width, r.y + r.height);
                assert(vec.rows == 1);
                assert(vec.cols == window_height * window_width * 16);
                //std::cout << vec << std::endl;
                ofs << "-1";
                for (unsigned int l = 0; l < vec.cols; ++l) {
                    ofs << "," << vec.at<float>(0,l);
                }
                ofs << std::endl;
                if (ofs_patch.is_open()) {
                    ofs_patch << "-1";
                    for (unsigned int l = 0; l < 24*12; l++) {
                        int x_coord = l % 24;
                        int y_coord = l / 24;

                        cv::Vec3b col = img.at<cv::Vec3b>(y+y_coord, x+x_coord);
                        ofs_patch << "," << int(col[0]) << "," << int(col[1]) << "," << int(col[2]);
                    }
                    ofs_patch << std::endl;
                }
                //cv::imshow("patch", img.rowRange(r.y, r.y + r.height).colRange(r.x, r.x + r.width));
                //cv::waitKey(0);
                break;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::string train_path;
    std::string gt_path;
    std::string output_file;
    std::string output_patch_file;
    int numbers = 0;
    int start = 100;
    int c;
    std::vector<int> excludes(nexcludes);
    excludes.assign(exclude_nums, exclude_nums + nexcludes);

    while ((c = getopt(argc, argv, "i:g:n:s:e:o:p:h")) != -1) {
        switch (c) {
            case 'i':
                train_path = optarg;
                break;
            case 'g':
                gt_path = optarg;
                break;
            case 'n':
                {
                std::stringstream stream(optarg);
                stream >> numbers;
                }
                break;
            case 's':
                {
                std::stringstream stream(optarg);
                stream >> start;
                }
                break;
            case 'e':
                excludes = spliti(std::string(optarg),',');
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'p':
                output_patch_file = optarg;
                break;
            case 'h':
            default:
                std::cout << c << std::endl;
                std::cerr << "Usage: extract_train_set OPTIONS" << std::endl 
                    << "\t -i <training directory>" << std::endl 
                    << "\t -g <gt box directory>" << std::endl 
                    << "\t -n <maximum id>" << std::endl 
                    << "\t -s <start id>" << std::endl
                    << "\t -o <output path>" << std::endl
                    << "\t -p <output patch file>" << std::endl
                    << "\t -e <exclude list>" << std::endl;
                return 1;
        }
    }

    if (train_path == "" || gt_path == "") {
        std::cerr << "Training path and gt path are mandatory" << std::endl;
        return 1;
    }
    if (output_file == "") {
        std::cerr << "output file is mandatory" << std::endl;
        return 1;
    }

    srand(time(NULL));

    std::ofstream ofs(output_file.c_str());

    std::ofstream ofs_patch;
    if (output_patch_file != "") {
        ofs_patch.open(output_patch_file);
    }

    for (unsigned int i = start; i <= numbers; ++i) {

        if (std::find(excludes.begin(), excludes.end(), i) != excludes.end()) continue;
        std::cout << "Processing: " << i << std::endl;

        std::string train_img;
        std::string gt_box;
        {
            std::stringstream stream(train_path);
            stream << train_path << "/" << i << ".jpg";
            train_img = (stream.str());
        }
        {
            std::stringstream stream(gt_path);
            stream << gt_path << "/" << i << ".txt";
            gt_box = (stream.str());
        }

        std::vector<cv::Rect> rects(read_boxes(gt_box));

        cv::Mat img = cv::imread(train_img);

        // first we extract the positive instances
        for (unsigned int j = 0; j < rects.size(); ++j) {
            extract_positive_samples(img, rects[j], ofs, ofs_patch);
        }

        std::cout << "Extracting random images" << std::endl;
        // then we gonna extract negative samples
        float size = img.rows;
        cv::Mat scaledImage = img;
        float scale = 1.0;
        for (unsigned int j = 0; j < nscales; ++j) {
            std::cout << "scale: " << size << std::endl;

            if (scaledImage.rows > (int) size) {
                float ratio = size / scaledImage.rows;
                int h = (int) size;
                int w = ratio * scaledImage.cols;
                std::cout << w << " " << h << " " << ratio << std::endl;
                cv::resize(img, scaledImage, cv::Size(w,h));
                rects = scale_rects(rects, ratio);
                scale = scale * ratio;
            }

            if (scaledImage.rows < window_height || scaledImage.cols < window_width) {
                continue;
            }

            extract_negative_samples(scaledImage, rects, scale, ofs, ofs_patch);

            size = size * scale_ratio;
        }
    }

    return 0;
}

