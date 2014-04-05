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
#include <text_detector/ProjectionProfileComputer.h>
#include <text_detector/RectShrinker.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

/**
 *  Returns the index of the box in which the point falls
 */
int find_box(const std::vector<cv::Rect> &boxes, const cv::Vec2f &pt)
{
    for (size_t i = 0; i < boxes.size(); i++) {
        if (pt[0] >= boxes[i].x && pt[0] <= boxes[i].x + boxes[i].width &&
            pt[1] >= boxes[i].y && pt[1] <= boxes[i].y + boxes[i].height)
            return i;
    }
    return -1;
}

static std::vector<cv::Rect> load_boxes(const fs::path &file)
{
    cv::TrainData data;
    data.read_csv(file.c_str());
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat samples = cv::Mat(data.get_values()).clone();

    std::vector<cv::Rect> result;
    result.reserve(samples.rows);
    for (int i = 0; i < samples.rows; i++) {
        result.push_back(cv::Rect(
            samples.at<float>(i,0),
            samples.at<float>(i,1),
            samples.at<float>(i,2),
            samples.at<float>(i,3)));
    }
    return result;
}

std::vector<int> load_textline_ids(const fs::path &file)
{
    cv::TrainData data;
    data.read_csv(file.generic_string().c_str());
    data.set_delimiter(',');
    cv::Mat line_idxs = cv::Mat(data.get_values()).clone();

    std::vector<int> results;
    results.resize(line_idxs.rows);
    for (int i = 0; i < line_idxs.rows; i++) {
        results[line_idxs.at<float>(i,0)] = line_idxs.at<float>(i,1);
    }
    return results;
}

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
        ("box-directory,b", po::value<std::string>(), "directory of the ground truth boxes")
        ("output,o", po::value<std::string>(), "output file of the features")
        ("show,s", "show training images")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    std::string directory, gt_directory, output_file,  box_directory;
    bool show = false;
    if (vm.count("directory")) 
        directory = vm["directory"].as<std::string>();
    if (vm.count("gt-directory")) 
        gt_directory = vm["gt-directory"].as<std::string>();
    if (vm.count("box-directory")) 
        box_directory = vm["box-directory"].as<std::string>();
    if (vm.count("output")) 
        output_file = vm["output"].as<std::string>();
    if (vm.count("show"))
        show = true;

    if (directory == "" || gt_directory == "" || output_file == "" || box_directory == "") {
        std::cout << desc << std::endl;
        return 1;
    }

    std::cout << "Directory: " << directory << " GT: " 
              << gt_directory << " output: " 
              << output_file << " Boxes: " << box_directory << std::endl;

    std::vector<std::string> dirs;
    dirs.push_back(directory); dirs.push_back(gt_directory);  dirs.push_back(box_directory);
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
        //labelled_image_path = "../train_icdar_2005_mser_cc/180";
        
        if (!fs::is_directory(labelled_image_path)) {
            std::cout << "Skipping: " << labelled_image_path << std::endl;
            continue;
        }

        fs::path train_img_no = labelled_image_path.filename();
        fs::path train_path = directory;
        train_path += "/";
        train_path += train_img_no;
        train_path += ".jpg";
        int imgno;
        std::stringstream(train_img_no.generic_string()) >> imgno;

        cv::Mat train_img = cv::imread(train_path.generic_string());
        std::cout << "Processing: " << train_path << std::endl;
        cv::Mat train_img_gray;
        cv::cvtColor(train_img, train_img_gray, CV_RGB2GRAY);
        cv::Mat gradient_img = TextDetector::compute_gradient(train_img);

        cv::Mat swt1, swt2;
        TextDetector::compute_swt(train_img_gray, swt1, swt2);

        std::vector<TextDetector::MserElement> image_elements;
        std::vector<cv::Mat> image_pairwise_features;
        for (fs::directory_iterator img_it(labelled_image_path); img_it != fs::directory_iterator(); ++img_it) {
            fs::path mser_root_element(*img_it);

            fs::path letters_path = mser_root_element;
            letters_path += "/letters.txt";
            if (fs::exists(letters_path)) {
                std::vector<TextDetector::MserElement> tmp(load_data(letters_path, 1));
                for (size_t i = 0; i < tmp.size(); i++) {
                    tmp[i].compute_features(train_img, gradient_img, swt1, swt2);
                }
                // filter overlapping features
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

                image_elements.insert(image_elements.end(), tmp.begin(), tmp.end());
            }

            fs::path neg_path = mser_root_element;
            neg_path += "/negatives.txt";
            if (fs::exists(neg_path)) {
                std::vector<TextDetector::MserElement> tmp(load_data(neg_path, -1));
                for (size_t i = 0; i < tmp.size(); i++) {
                    tmp[i].compute_features(train_img, gradient_img, swt1, swt2);
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

        std::vector<int> text_line_ids;
        std::map<int,std::vector<int> > inverted_text_line_ids;
        fs::path textline_id_path = labelled_image_path;
        textline_id_path += "/text_lines.txt";
        if (fs::exists(textline_id_path)) {
            text_line_ids = load_textline_ids(textline_id_path);
            // create an inverted index from the text lines
            for (int i = 0; i < text_line_ids.size(); i++) {
                int text_line_id = text_line_ids[i];
                int uid = i;
                int id = -1;
                for (size_t j = 0; j < image_elements.size(); j++) {
                    if (image_elements[j].get_uid() == uid) {
                        id = j;
                        break;
                    }
                }
                if (id < 0) {
                    continue;
                }
                if (inverted_text_line_ids.find(text_line_id) == inverted_text_line_ids.end()) {
                    inverted_text_line_ids.insert(std::make_pair(text_line_id, std::vector<int>()));
                }
                inverted_text_line_ids[text_line_id].push_back(id);
            }
        } else {
            std::vector<int> uids(image_elements.size());
            std::transform(image_elements.begin(), image_elements.end(), uids.begin(), [](const TextDetector::MserElement &el) { return el.get_uid(); });
            int the_max = *std::max_element(uids.begin(), uids.end());
            text_line_ids = std::vector<int>(the_max + 1, -1);
        }

        // do the pairwise stuff
        std::vector<int> num_neighbors(image_elements.size(), 0);
        // all neighbors is an adjacency list. The first element of the pair represents the distance to the element, 
        // the second the id
        std::vector<std::vector<std::pair<float, int> > > all_neighbors(image_elements.size());

        // load the GT-boxes
        fs::path gt_box_path = box_directory;
        gt_box_path += "/";
        gt_box_path += train_img_no;
        gt_box_path += ".txt";

        std::vector<cv::Rect> boxes;
        if (fs::exists(gt_box_path)) {
            boxes = load_boxes(gt_box_path);
        }

        // fill neighbors if elements are in the same textline
        for (size_t i = 0; i < image_elements.size(); i++) {
            cv::Vec2f c1 = image_elements[i].get_centroid();
            cv::Rect r1 = image_elements[i].get_bounding_rect();
            if (image_elements[i].get_label() <= 0) continue;
            for (size_t j = 0; j < image_elements.size(); j++) {
                if (i == j) continue;
                if (image_elements[j].get_label() <= 0) continue;

                cv::Vec2f c2 = image_elements[j].get_centroid();

                if (find_box(boxes, c1) == find_box(boxes, c2) || 
                    text_line_ids[image_elements[i].get_uid()] == text_line_ids[image_elements[j].get_uid()]) {
                    all_neighbors[i].push_back(
                        std::make_pair(1.0f, j));
                }
            }
        }
        std::vector<std::vector<bool> > adjacency_matrix;
        adjacency_matrix.resize(image_elements.size());
        for (size_t i = 0; i < adjacency_matrix.size(); i++) {
            adjacency_matrix[i].resize(image_elements.size());
        }

        std::vector<int> idxs(all_neighbors.size());
        for (int i = 0; i < idxs.size(); i++) idxs[i] = i;

        std::sort(idxs.begin(), idxs.end(), [&image_elements] (int a, int b) -> bool {
            return image_elements[a].get_centroid()[0] < image_elements[b].get_centroid()[0];
        });

        // link the features
        for (int i : idxs) {
        //for (int i = 0; i < all_neighbors.size(); i++) {
            if (image_elements[i].get_label() <= 0) continue;
            auto neighbors = all_neighbors[i];
            float x_coord = image_elements[i].get_centroid()[0];

            neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(), [x_coord, &image_elements] (const std::pair<float, int> &p) -> bool {
                return image_elements[p.second].get_centroid()[0] < x_coord;
            }), neighbors.end());

            std::sort(neighbors.begin(),
                      neighbors.end(),
                      [&image_elements] (const std::pair<float, int> &p1,
                                         const std::pair<float, int> &p2) -> bool {
                return image_elements[p1.second].get_centroid()[0] < image_elements[p2.second].get_centroid()[0];
            });

            //cv::Mat img = train_img.clone();
            //image_elements[i].draw(img);
            //for (auto pair : neighbors) {
            //    image_elements[pair.second].draw(img);
            //    break;
            //}
            //cv::imshow("PAIRS", img);
            //cv::waitKey(0);

            for (int j = 0; j < neighbors.size(); j++) {
                int id = neighbors[j].second;

                if (adjacency_matrix[id][i] || adjacency_matrix[i][id]) continue;

                adjacency_matrix[i][id] = true;
                adjacency_matrix[id][i] = true;


                std::vector<int> element_idxs = inverted_text_line_ids[text_line_ids[image_elements[id].get_uid()]];

                int tl = INT_MAX;
                int br = INT_MIN;
                for (int idx : element_idxs) {
                    tl = std::min(tl, image_elements[idx].get_bounding_rect().tl().x);
                    br = std::max(br, image_elements[idx].get_bounding_rect().br().x);
                }
                assert(element_idxs.size() > 0);

                TextDetector::ProjectionProfileComputer pp(cv::Size(br - tl + 1, 0), tl);
                cv::Mat pps = pp.compute(image_elements[element_idxs[0]].get_pixels());
                float mean_height = image_elements[element_idxs[0]].get_bounding_rect().height;
                for (int k = 1; k < element_idxs.size(); k++) {
                    pps = pp.compute(image_elements[element_idxs[k]].get_pixels(), pps);
                    mean_height += image_elements[element_idxs[k]].get_bounding_rect().height;
                }
                mean_height /= element_idxs.size();

                int thresh = pp.compute_threshold(pps);
                cv::Mat gaps = pps < thresh;

                TextDetector::RectShrinker shrinker(0.10, tl);
                std::vector<cv::Rect> rects; 
                rects.push_back(image_elements[i].get_bounding_rect());
                rects.push_back(image_elements[id].get_bounding_rect());
                rects = shrinker.shrink(rects, gaps);
                
                float dx1 = image_elements[id].get_bounding_rect().tl().x - image_elements[i].get_bounding_rect().br().x;
                float dx2 = rects[1].tl().x - rects[0].br().x;
                float dx3 = image_elements[id].get_centroid()[0] - image_elements[i].get_centroid()[0];

                if (dx2 <= 0 || dx1 <= 0 || dx3 <= 0) {
                    std::cout << "imgno: " << imgno <<  " text-line-id: " << text_line_ids[image_elements[i].get_uid()] 
                        << " DX1: " << dx1 << " DX2: " << dx2 << " DX3: " << dx3  << std::endl;
                    break;
                }

                // imageno, textline-id, label, distance
                cv::Mat feature(1, 9, CV_32FC1, cv::Scalar(0));

                feature.at<float>(0,0) = imgno;
                feature.at<float>(0,1) = image_elements[i].get_uid();
                feature.at<float>(0,2) = image_elements[id].get_uid();
                feature.at<float>(0,3) = text_line_ids[image_elements[i].get_uid()];
                feature.at<float>(0,4) = find_box(boxes, image_elements[i].get_centroid()) == find_box(boxes, image_elements[id].get_centroid()) ? 1.0 : -1.0;
                feature.at<float>(0,5) = dx1;
                feature.at<float>(0,6) = dx2;
                feature.at<float>(0,7) = dx3;
                feature.at<float>(0,8) = mean_height;

                // only next neighbor!
                pairwise_features.push_back(feature);
                image_pairwise_features.push_back(feature);
                break;
            }
        }

        if (show) {
            cv::Mat img = train_img.clone();
            std::map<int, int> uid_to_idx;
            for (size_t i = 0; i < image_elements.size(); i++) {
                uid_to_idx[image_elements[i].get_uid()] = i;
                image_elements[i].draw(img);
            }
            for (size_t i = 0; i < image_pairwise_features.size(); i++) {
                int uid1 = image_pairwise_features[i].at<float>(0,1);
                int uid2 = image_pairwise_features[i].at<float>(0,2);
                int idx1 = uid_to_idx[uid1];
                int idx2 = uid_to_idx[uid2];

                cv::Point p1(
                    image_elements[idx1].get_centroid()[0],
                    image_elements[idx1].get_centroid()[1]);
                cv::Point p2(
                    image_elements[idx2].get_centroid()[0],
                    image_elements[idx2].get_centroid()[1]);
                cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1, 8);
            }
            cv::imshow("ABC", img);
            cv::waitKey(0);
        }
        elements.insert(elements.end(), image_elements.begin(), image_elements.end());
    }

    std::cout << "Elements: " << elements.size() << " PW: " << pairwise_features.size() << std::endl;
    {
        std::ofstream ofs(output_file.c_str());
        for (size_t i = 0; i < pairwise_features.size(); i++) {

            for (int j = 0; j < pairwise_features[i].cols; j++) {
                ofs << pairwise_features[i].at<float>(0,j);
                if (j != pairwise_features[i].cols - 1)
                    ofs << ",";
            }
            ofs << std::endl;
        }
    }
    std::cout << "Wrote data" << std::endl;

    return 0;
}
