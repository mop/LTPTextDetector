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
#include <getopt.h>
#include <sstream>

/**
 * This is a helper program used for training an adaboost variant from opencv.
 */
int main(int argc, char *argv[])
{
    int c;
    std::string train_file;
    std::string output_file;
    int nweak_learners = 100;
    int ndepth = 5;
    int boost_type = cv::Boost::GENTLE;
    float prior = 1.0f;
    while ((c = getopt(argc, argv, "i:o:n:d:v:p:h")) != -1) {
        switch (c) {
            case 'i':
                train_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'n':
                std::stringstream(optarg) >> nweak_learners;
                break;
            case 'd':
                std::stringstream(optarg) >> ndepth;
                break;
            case 't':
                std::stringstream(optarg) >> boost_type;
                break;
            case 'p':
                std::stringstream(optarg) >> prior;
                break;
            case 'h':
            default:
                std::cerr << "Usage: train_forest OPTIONS" << std::endl 
                    << "\t -i <train.csv>" << std::endl 
                    << "\t -o <out.txt>" << std::endl 
                    << "\t -n <number of weak classifiers>" << std::endl 
                    << "\t -t <boosting type>" << std::endl 
                    << "\t -p <prior/weight of text>" << std::endl 
                    << "\t -d <depth of trees>" << std::endl;
                return 1;
        }
    }
    if (ndepth < 1) { 
        std::cerr << "Error - depth of trees must be at least 1" << std::endl;
        return 1;
    }
    if (nweak_learners < 1) {
        std::cerr << "Error - need at least 1 learner" << std::endl;
        return 1;
    }
    if (boost_type < 0 || boost_type > 3) {
        std::cerr << "Error - invalid boosting type" << std::endl;
        return 1;
    }
    if (train_file == "") {
        std::cerr << "Error - need a training file" << std::endl;
        return 1;
    }
    if (output_file == "") {
        std::cerr << "Error - need an output file" << std::endl;
        return 1;
    }
    std::cout << train_file << std::endl;
    std::cout << output_file << std::endl;
    std::cout << nweak_learners << " " << ndepth << " " << boost_type << std::endl;

    cv::TrainData data;
    data.read_csv(train_file.c_str());
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat values = cv::Mat(data.get_values());
    cv::Mat samples = values.colRange(1, values.cols);
    cv::Mat labels = values.colRange(0, 1);

    std::cout << labels.rows << " " << labels.cols << std::endl;
    std::cout << samples.rows << " " << samples.cols << std::endl;

    cv::Boost boost;
    float priors[] = { 1.0f, prior };
    cv::Mat vartype(samples.cols+1, 1, CV_8U);
    vartype.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    vartype.at<uchar>(samples.cols, 0) = CV_VAR_CATEGORICAL;
    boost.train(samples, CV_ROW_SAMPLE, labels, cv::Mat(), cv::Mat(), vartype, cv::Mat(), CvBoostParams(
        boost_type,          // boost type
        nweak_learners,      // # weak learners
        0.0,                 // trim rate?!
        ndepth,              // depth of weak classifier
        false,               // use surrogates
        priors               // priors
    ));
    cv::FileStorage fs(output_file.c_str(), cv::FileStorage::WRITE);
    boost.write(*fs, "trees");
    return 0;
}

