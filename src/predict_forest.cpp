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
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
#include <getopt.h>
#include <sstream>

static std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

static std::vector<int> split(const std::string &s)
{
    std::vector<std::string> parts = split(s, ':');
    std::vector<int> results;
    for (int i = 0; i < parts.size(); ++i) {
        std::stringstream stream(parts[i]);
        int tmp;
        stream >> tmp;
        results.push_back(tmp);
    }
    return results;
}

static void parse_steps(const std::string &arg, int &start, int &step, int &end)
{
    std::string tmp(optarg);
    std::vector<int> results = split(tmp);
    if (results.size() >= 1) {
        start = results[0];
    }
    if (results.size() >= 2) {
        step = results[1];
    }
    if (results.size() >= 3) {
        end = results[2];
    }
}

/**
 * This is a helper program for cross-validating random forests.
 * The training file should be randomized. Note: cross-validation is actually
 * not really necessary to do when using Random Forests, since they have 
 * an own cross-validation like error-measure: oob-error
 */
int main(int argc, char *argv[])
{
    int c;
    std::string train_file;
    std::string model_file;
    std::string out_file;
    while ((c = getopt(argc, argv, "i:o:m:h")) != -1) {
        switch (c) {
            case 'i':
                train_file = optarg;
                break;
            case 'o':
                out_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'h':
            default:
                std::cerr << "Usage: train_forest OPTIONS" << std::endl 
                    << "\t -i <train.csv>" << std::endl 
                    << "\t -o <out.csv>" << std::endl 
                    << "\t -m <model-file.yml>" << std::endl ;
                return 1;
        }
    }
    if (train_file == "" || model_file == "" || out_file == "") {
        std::cerr << "Error - need a training, output and model file" << std::endl;
        return 1;
    }
    std::cout << train_file << " " << model_file << " " << out_file << std::endl;

    cv::TrainData data;
    data.read_csv(train_file.c_str());
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat values = cv::Mat(data.get_values());

    cv::RandomTrees rf;
    cv::FileStorage fs(model_file.c_str(), cv::FileStorage::READ);
    rf.read(*fs, *fs["trees"]);

    std::cout << values.rows << std::endl;
    std::ofstream ofs(out_file.c_str());
    for (int i = 0; i < values.rows; i++) {
        ofs << rf.predict_prob(values.row(i)) << std::endl;
    }
    return 0;
}
