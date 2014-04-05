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

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>

#include <text_detector/config.h>
typedef dlib::matrix<double,0,1> vector_type;


namespace po = boost::program_options;
namespace fs = boost::filesystem;

static void construct_maps(
    const cv::Mat &train_data, const cv::Mat &pw_data, 
    std::vector<std::map<int, int> > &uid_to_index, 
    std::vector<std::map<int, std::vector<int> > > &uid_to_pw_index)
{
    double min, max;
    cv::minMaxLoc(train_data.col(0), &min, &max);

    // fetch the maximum number of different images to resize index-vectors
    uid_to_index.resize(max+1);
    uid_to_pw_index.resize(max+1);

    for (int i = 0; i < train_data.rows; i++) {
        int img_id = train_data.at<float>(i,0);
        int uid = train_data.at<float>(i,1);
        uid_to_index[img_id][uid] = i;  // store the index in the hashtable
    }

    for (int i = 0; i < pw_data.rows; i++) {
        int img_id = pw_data.at<float>(i,0);
        int uid1 = pw_data.at<float>(i,1);
        int uid2 = pw_data.at<float>(i,2);
        uid_to_pw_index[img_id][uid1].push_back(i);
        //uid_to_pw_index[img_id][uid2].push_back(i);       // only need to add one link!
    }
}

static cv::Mat read_csv(const std::string &filename)
{
    std::ifstream ifs(filename.c_str());
    std::string line;
    std::vector<std::vector<float> > data;

    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string part;
        std::vector<float> row;
        while (std::getline(ss, part, ',')) {
            float val;
            std::stringstream(part) >> val;
            row.push_back(val);
        }
        data.push_back(row);
    }

    cv::Mat result(data.size(), data[0].size(), CV_32FC1);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at<float>(i,j) = data[i][j];
        }
    }
    return result;
}

int main(int argc, const char *argv[])
{
    srand(time(NULL));
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "print this help message")
        ("input,i", po::value<std::string>(), "the input csv file")
        ("input-pairwise,p", po::value<std::string>(), "the pairwise-input file")
        ("model,m", po::value<std::string>(), "the model file")
        ("output,o", po::value<std::string>(), "output file of the model")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    std::string input_file, input_pairwise_file, output_file, model_file;
    if (vm.count("input")) 
        input_file = vm["input"].as<std::string>();
    if (vm.count("input-pairwise")) 
        input_pairwise_file = vm["input-pairwise"].as<std::string>();
    if (vm.count("output")) 
        output_file = vm["output"].as<std::string>();
    if (vm.count("model")) 
        model_file = vm["model"].as<std::string>();

    if (input_file == "" || input_pairwise_file == "" || output_file == "" || model_file == "") {
        std::cout << desc << std::endl;
        return 1;
    }

    if (!fs::exists(input_file) || !fs::exists(input_pairwise_file) || !fs::exists(model_file)) {
        std::cout << "Error, one of the input files does not exist" << std::endl;
        return 1;
    }

    std::ifstream ifs(model_file.c_str(), std::ios::binary);
    dlib::graph_labeler<vector_type> labeler;
    dlib::deserialize(labeler, ifs);

    cv::Mat input = read_csv(input_file);
    cv::Mat pairwise = read_csv(input_pairwise_file);

    std::cout << "read: " << input.rows << " unary features and " << pairwise.rows << " pairwise features" << std::endl;
    std::cout << "unary dimension: " << input.cols << " " << " pw dimension: " << pairwise.cols << std::endl;

    std::vector<std::map<int, int> > uid_to_index;
    std::vector<std::map<int, std::vector<int> > > uid_to_pairwise_index;
    construct_maps(input, pairwise, uid_to_index, uid_to_pairwise_index);

    std::ofstream ofs(output_file.c_str());

    // foreach training image
    for (size_t i = 0; i < uid_to_index.size(); i++) {
        // create a graph
        if (uid_to_index[i].empty()) continue; // skip the image...

        size_t n_nodes = uid_to_index[i].size();

        graph_type graph;
        graph.set_number_of_nodes(n_nodes);

        std::map<int, int> uid_to_var;  // we need this for the pw model
        std::map<int, int> var_to_uid;  // we need this for the result

        size_t j = 0; // stores the varname
        for (auto it = uid_to_index[i].begin(); it != uid_to_index[i].end(); ++it, ++j) {
            int uid = it->first;
            int idx = it->second;

            // create the unary factor
            node_vector_type data;
            data(0,0) = 1.0;
            for (int k = 2; k < input.cols; k++) {
                data(k-2+1, 0) = input.at<float>(idx, k);
            }

            graph.node(j).data = data;

            uid_to_var.insert(std::make_pair(uid, j));
            var_to_uid.insert(std::make_pair(j, uid));
        }

        // now create the pairwise factors...
        // walk through all the pairwise factors for the given image
        for (auto it = uid_to_pairwise_index[i].begin(); it != uid_to_pairwise_index[i].end(); ++it) {
            int uid = it->first;
            if (uid_to_var.find(uid) == uid_to_var.end()) {
                std::cerr << "could not find uid: " << uid << " in image: " << i << std::endl;
                continue;
            }
            int v1 = uid_to_var[uid];
            std::vector<int> pw_idxs = it->second;
            // for all adjacent vertices... -> create a pairwise factor
            for (size_t k = 0; k < pw_idxs.size(); k++) {
                int other_uid = pairwise.at<float>(pw_idxs[k], 2);

                // get the variable names...
                if (uid_to_var.find(other_uid) == uid_to_var.end()) {
                    std::cerr << "could not find uid: " << uid << " in image: " << i << std::endl;
                    continue;
                }
                int v2 = uid_to_var[other_uid];
                if (graph.has_edge(v1, v2)) {
                    std::cerr << "GRAPH HAS EDGE: " << v1 << " " << v2 << std::endl;
                    continue;
                }
                graph.add_edge(v1, v2);

                edge_vector_type data;
                
                data(0,0) = 1.0;
                for (int l = 3; l < pairwise.cols; l++) {
                    data(l-3+1, 0) = pairwise.at<float>(pw_idxs[k], l);
                }
                dlib::edge(graph, v1, v2) = data;
            }
        }

        std::vector<bool> labels = labeler(graph);
        for (size_t k = 0; k < labels.size(); k++) {
            int uid = var_to_uid[k];
            ofs << i << "," << uid << "," << labels[k] << std::endl;
        }
    }
    ofs.close();
    input.release();
    pairwise.release();
    return 0;
}

