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
#include <text_detector/HierarchicalMSER.h>
#include <text_detector/MserTree.h>
#include <text_detector/LabelWidget.h>

#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <QApplication>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

static std::vector<int> read_gt(const std::string &path)
{
    std::vector<int> result;
    std::cout << path << std::endl;
    std::ifstream ifs(path.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
        int tmp;
        std::stringstream(line) >> tmp;
        result.push_back(tmp);
    }
    return result;
}

static std::vector<std::vector<cv::Point> > load_contours(const std::string &path, const std::vector<int> &ids)
{
    std::vector<std::vector<cv::Point> > result;
    for (size_t i = 0; i < ids.size(); i++) {
        std::stringstream ss;
        ss << path << "/node" << ids[i] << "_contour.csv";

        std::string s(ss.str());

        cv::TrainData data;
        data.read_csv(s.c_str());
        std::cout << s << std::endl;
        data.set_delimiter(',');
        cv::Mat vals = data.get_values();
        std::cout << "Loaded mat with: " << vals.rows << " " << vals.cols << std::endl;

        std::vector<cv::Point> contour;
        contour.reserve(vals.rows);
        for (int j = 0; j < vals.rows; j++) {
            contour.push_back(cv::Point(int (vals.at<float>(j,0)), int (vals.at<float>(j,1))));
        }
        result.push_back(contour);
    }
    return result;
}

static void write_letters(const std::string &path, const std::vector<int> &uids)
{
    std::ofstream ofs(path);
    for (size_t i = 0; i < uids.size(); i++) {
        ofs << uids[i] << std::endl;
    }
    
}

static cv::Mat to_mat(const std::vector<cv::Point> &contour, const cv::Size &size)
{
    cv::Mat result(size.height, size.width, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < contour.size(); i++) {
        result.at<unsigned char>(contour[i].y, contour[i].x) = 255;
    }
    return result;
}

static void fixup_msers(const std::string &train_directory, const std::string &output_directory)
{
    bool skip = true;
    for (fs::directory_iterator it = fs::directory_iterator(output_directory); it != fs::directory_iterator(); ++it) {
        std::string number = fs::path(*it).stem().generic_string();
        std::cout << "Processing: " << number << std::endl;
        if (number == "111") {
            skip = false;
            continue;
        }

        if (skip) continue;
        if (!fs::is_directory(fs::path(*it))) continue;
        
        fs::path train_image_path = train_directory;
        train_image_path += "/";
        train_image_path += number;
        train_image_path += ".jpg";

        cv::Mat train_image = cv::imread(train_image_path.generic_string());
        cv::Mat gray_train_image;
        cv::cvtColor(train_image, gray_train_image, CV_RGB2GRAY);

        cv::HierarchicalMSER mser;
        std::vector<double> vars;
        std::vector<std::vector<cv::Point> > msers;
        std::vector<cv::Vec4i> hierarchy;
        mser(gray_train_image, msers, vars, hierarchy);
        TextDetector::MserTree tree(msers, vars, hierarchy);

        // search each gt-letter thing
        for (fs::directory_iterator subit = fs::directory_iterator(*it); subit != fs::directory_iterator(); ++subit) {
            fs::path letters_path = fs::path(*subit);
            letters_path += "/letters.txt";
            if (!fs::exists(letters_path)) continue;

            std::vector<int> label_list = read_gt(letters_path.generic_string());
            std::vector<std::vector<cv::Point> > contour_list = load_contours(fs::path(*subit).generic_string(), label_list);
            std::vector<int> new_ids;
            for (size_t i = 0; i < contour_list.size(); i++) {
                cv::Mat mask = to_mat(contour_list[i], cv::Size(train_image.cols, train_image.rows));
                int new_id = tree.match_contour(mask);
                std::cout << "Matched contour -> new idx: " << new_id << std::endl;
                if (new_id >= 0) {
                    new_ids.push_back(new_id);
                }
            }

            write_letters(letters_path.generic_string(), new_ids);
        }
    }
}


int main(int argc, char *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "print this help message")
        ("directory,i", po::value<std::string>(), "directory of the image files")
        ("gt-directory,g", po::value<std::string>(), "directory of the ground truth files")
        ("output,o", po::value<std::string>(), "output directory of the mser ccs")
        ("fixup,f", "fixup gt msers by stored contours")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    bool fixup = false;
    std::string directory, gt_directory, output_directory;
    if (vm.count("directory")) 
        directory = vm["directory"].as<std::string>();
    if (vm.count("gt-directory")) 
        gt_directory = vm["gt-directory"].as<std::string>();
    if (vm.count("output")) 
        output_directory = vm["output"].as<std::string>();
    if (vm.count("fixup")) 
        fixup = true;

    if (directory == "" || gt_directory == "" || output_directory == "") {
        std::cout << desc << std::endl;
        return 1;
    }

    std::cout << "Directory: " << directory << " GT: " << gt_directory << " output: " << output_directory << std::endl;

    std::vector<std::string> dirs;
    dirs.push_back(directory); dirs.push_back(gt_directory); dirs.push_back(output_directory);
    for (auto it = dirs.begin(); it != dirs.end(); ++it) {
        if (!fs::is_directory(*it)) {
            std::cerr << *it << " is not a directory" << std::endl;
            return 1;
        }
    }

    if (fixup) {
        std::cout << "Fixing GT, please wait" << std::endl;
        fixup_msers(directory, output_directory);
    }

    QApplication app(argc, argv);
    LabelWidget label_widget(directory, gt_directory, output_directory);

    label_widget.show();
    return app.exec();
}
