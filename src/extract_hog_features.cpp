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
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <text_detector/CCUtils.h>
#include <text_detector/config.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

static std::vector<TextDetector::MserElement> load_data(const fs::path &file, int label)
{
    fs::path dirname(file.parent_path());
    std::string imgid_path = dirname.parent_path().stem().generic_string();
    int imgid;
    std::stringstream(imgid_path) >> imgid;
    dirname += "/";
    std::vector<TextDetector::MserElement> result;
    std::ifstream ifs(file.generic_string().c_str());
    std::string line;

    while (std::getline(ifs, line)) {
        std::stringstream filename;
        filename << "node" << line << "_contour.csv";
        std::string fname(filename.str());
        fs::path contour_path = dirname;
        contour_path += fname;

        std::vector<cv::Point> pixels(TextDetector::load_pixels(contour_path));

        int uid;
        std::stringstream(line) >> uid;

        result.push_back(TextDetector::MserElement(imgid, uid, label, pixels));
    }

    return result;
}

int main(int argc, const char *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "print this help message")
        ("directory,i", po::value<std::string>(), "directory of the image files")
        ("gt-directory,g", po::value<std::string>(), "directory of the ground truth files")
        ("output,o", po::value<std::string>(), "output file of the features")
        ("output-binary,b", po::value<std::string>(), "output file of the binary masks")
        ("filter-overlapping,f", "filter overlapping features")
        ("show,s", "show training images")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    std::string directory, gt_directory, output_file, output_binary_file;
    bool filter_overlapping = false;
    bool show = false;
    if (vm.count("directory")) 
        directory = vm["directory"].as<std::string>();
    if (vm.count("gt-directory")) 
        gt_directory = vm["gt-directory"].as<std::string>();
    if (vm.count("output")) 
        output_file = vm["output"].as<std::string>();
    if (vm.count("output-binary")) 
        output_binary_file = vm["output-binary"].as<std::string>();
    if (vm.count("filter-overlapping"))
        filter_overlapping = true;
    if (vm.count("show"))
        show = true;

    if (directory == "" || gt_directory == "" || output_file == "") {
        std::cout << desc << std::endl;
        return 1;
    }

    std::cout << "Directory: " << directory << " GT: " 
              << gt_directory << " output: " 
              << output_file << " output-binary: " << output_binary_file
              << " filter overlapping: " << filter_overlapping << std::endl;

    std::vector<std::string> dirs;
    dirs.push_back(directory); dirs.push_back(gt_directory); 
    for (auto it = dirs.begin(); it != dirs.end(); ++it) {
        if (!fs::is_directory(*it)) {
            std::cerr << *it << " is not a directory" << std::endl;
            return 1;
        }
    }

    fs::path labelled_data_root(gt_directory);
    std::vector<TextDetector::MserElement> elements;
    std::vector<cv::Mat> pairwise_features;

    for (fs::directory_iterator it(labelled_data_root); it != fs::directory_iterator(); ++it) {
        fs::path labelled_image_path(*it);
        
        if (!fs::is_directory(labelled_image_path)) {
            std::cout << "Skipping: " << labelled_image_path << std::endl;
            continue;
        }

        fs::path train_img_no = fs::path(*it).filename();
        fs::path train_path = directory;
        train_path += "/";
        train_path += train_img_no;
        train_path += ".jpg";
        int imgno;
        std::stringstream(train_img_no.generic_string()) >> imgno;

        cv::Mat train_img = cv::imread(train_path.generic_string());
        std::cout << "Processing: " << train_path << std::endl;
        if (train_img.depth() > 3) {
            cv::cvtColor(train_img, train_img, CV_RGBA2RGB);
        }
        cv::Mat train_img_gray;
        cv::cvtColor(train_img, train_img_gray, CV_RGB2GRAY);
        //cv::Mat gradient_img = TextDetector::compute_hogs(train_img);
        cv::Mat gradient_img = TextDetector::compute_gradient(train_img);
        //ImagePyramid<HogIntegralImageComputer> hog_computer(10, 2/3.0, 12, train_img_gray);
        //hog_computer.set_image(train_img_gray);

        std::vector<TextDetector::MserElement> image_elements;
        for (fs::directory_iterator img_it(labelled_image_path); img_it != fs::directory_iterator(); ++img_it) {
            fs::path mser_root_element(*img_it);

            fs::path letters_path = mser_root_element;
            letters_path += "/letters.txt";

            if (fs::exists(letters_path)) {
                std::vector<TextDetector::MserElement> tmp(load_data(letters_path, 1));
                for (size_t i = 0; i < tmp.size(); i++) {
                    tmp[i].compute_hog_features(train_img_gray);
                }
                // filter overlapping features
                if (filter_overlapping) {
                    std::set<int> to_remove;
                    for (size_t i = 0; i < tmp.size(); i++) {
                        cv::Rect rect1 = tmp[i].get_bounding_rect();
                        for (size_t j = i+1; j < tmp.size(); j++) {
                            cv::Rect rect2 = tmp[j].get_bounding_rect();
                            cv::Rect intersect = rect1 & rect2;
                            if (intersect.width * intersect.height > 0.8 * std::min(rect1.width * rect1.height, rect2.width * rect2.height)) {
                                int smaller_idx = rect1.width * rect1.height < rect2.width * rect2.height ? tmp[i].get_uid() : tmp[j].get_uid();
                                to_remove.insert(smaller_idx);
                            }
                        }
                    }

                    tmp.erase(std::remove_if(tmp.begin(), tmp.end(), [&to_remove](const TextDetector::MserElement &el) -> bool {
                        return to_remove.find(el.get_uid()) != to_remove.end();
                    }), tmp.end());

                }
                image_elements.insert(image_elements.end(), tmp.begin(), tmp.end());
            }

            fs::path neg_path = mser_root_element;
            neg_path += "/negatives.txt";

            if (fs::exists(neg_path)) {
                std::vector<TextDetector::MserElement> tmp(load_data(neg_path, -1));
                for (size_t i = 0; i < tmp.size(); i++) {
                    //tmp[i].compute_features(train_img, gradient_img, hog_computer, swt1, swt2);
                    tmp[i].compute_hog_features(train_img_gray);
                }
                image_elements.insert(image_elements.end(), tmp.begin(), tmp.end());
            }
        }

        std::sort(image_elements.begin(), image_elements.end(), [](const TextDetector::MserElement &el1, const TextDetector::MserElement &el2) -> bool {
            return el1.get_uid() < el2.get_uid();
        });
        image_elements.erase(std::unique(image_elements.begin(), image_elements.end(), [](const TextDetector::MserElement &el1, const TextDetector::MserElement &el2) -> bool {
            return el1.get_uid() == el2.get_uid();
        }), image_elements.end());
        std::sort(image_elements.begin(), image_elements.end(), [](const TextDetector::MserElement &el1, const TextDetector::MserElement &el2) -> bool {
            return el1.get_uid() < el2.get_uid();
        });
        
        elements.insert(elements.end(), image_elements.begin(), image_elements.end());
    }

    std::cout << "Elements: " << elements.size() << " PW: " << pairwise_features.size() << std::endl;

    cv::Mat data(elements.size(), 1 + elements[0].get_hog_features().cols, CV_32FC1);
    cv::Mat binary_data(elements.size(), 1 + 28*28, CV_32FC1);

    for (size_t i = 0; i < elements.size(); i++) {
        data.at<float>(i,0)  = elements[i].get_label();
        cv::Mat vec = elements[i].get_hog_features();
        cv::Mat submat = data.rowRange(i, i+1).colRange(1, 1+vec.cols);
        vec.copyTo(submat);

        binary_data.at<float>(i,0) = elements[i].get_label();
        cv::Mat bin_vec = elements[i].get_binary_image();
        cv::Mat bin_submat = binary_data.rowRange(i, i+1).colRange(1, 1+28*28);
        bin_vec.convertTo(bin_vec, CV_32FC1);
        bin_vec.copyTo(bin_submat);
    }

    std::cout << "Copied data" << std::endl;

    {
        std::ofstream ofs(output_file.c_str());
        for (int i = 0; i < data.rows; i++) {
            for (int j = 0; j < data.cols; j++) {
                ofs << data.at<float>(i,j);
                if (j != data.cols - 1)
                    ofs << ",";
            }
            ofs << std::endl;
        }
    }
    {
        std::ofstream ofs(output_binary_file.c_str());
        for (int i = 0; i < binary_data.rows; i++) {
            for (int j = 0; j < binary_data.cols; j++) {
                ofs << binary_data.at<float>(i,j);
                if (j != binary_data.cols - 1)
                    ofs << ",";
            }
            ofs << std::endl;
        }
    }

    std::cout << "Wrote data" << std::endl;

    return 0;
}
