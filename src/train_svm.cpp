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

static double to_double(const std::string &str)
{
    double result = 0.0;
    std::stringstream ss(str);
    ss >> result;
    return result;
}

/**
 * This is a helper program used for training an adaboost variant from opencv.
 */
int main(int argc, char *argv[])
{
    int c;
    std::string train_file;
    std::string output_file;
    double C = 1;
    int k = 10;
    int boost_type = cv::Boost::GENTLE;
    bool auto_train = true;
    float prior = 1.0f;
    float gamma = 0.5f;
    bool use_rbf = false;
    CvParamGrid grid = CvSVM::get_default_grid(CvSVM::C);
    while ((c = getopt(argc, argv, "r:s:i:o:c:p:a:k:g:h")) != -1) {
        switch (c) {
            case 'i':
                train_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'r':
                use_rbf = true;
                break;
            case 'c':
                {
                std::stringstream stream(optarg);
                stream >> C;
                }
                break;
            case 's':
                {
                std::stringstream stream(optarg);
                stream >> gamma;
                }
                break;
            case 'p':
                {
                std::stringstream stream(optarg);
                stream >> prior;
                }
                break;
            case 'a':
                {
                std::stringstream stream(optarg);
                stream >> auto_train;
                }
                break;
            case 'k':
                {
                std::stringstream stream(optarg);
                stream >> k;
                }
                break;
            case 'g':
                {
                std::stringstream stream(optarg);
                std::string start;
                std::string step;
                std::string stop;
                std::getline(stream, start, ':');
                std::getline(stream, step, ':');
                std::getline(stream, stop, ':');

                grid.min_val = to_double(start);
                grid.step = to_double(step);
                grid.max_val = to_double(stop);
                }
                break;
            case 'h':
            default:
                std::cerr << "Usage: train_svm OPTIONS" << std::endl 
                    << "\t -i <train.csv>" << std::endl 
                    << "\t -o <out.txt>" << std::endl 
                    << "\t -c <C>" << std::endl 
                    << "\t -a <train auto>" << std::endl 
                    << "\t -k <k folds>" << std::endl 
                    << "\t -s <gamma>" << std::endl 
                    << "\t -r <use rbf>" << std::endl 
                    << "\t -p <prior/weight of text>" << std::endl;
                return 1;
        }
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
    std::cout << C << std::endl;

    cv::TrainData data;
    data.read_csv(train_file.c_str());
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat values = cv::Mat(data.get_values());
    cv::Mat samples = values.colRange(1, values.cols);
    cv::Mat labels = values.colRange(0, 1);

    std::cout << labels.rows << " " << labels.cols << std::endl;
    std::cout << samples.rows << " " << samples.cols << std::endl;
    std::cout << samples.at<float>(0,0) << std::endl;


    cv::SVM svm;
    //float priors[] = { 1.0f, prior };
    cv::Mat class_weights(1,2,CV_32FC1, cv::Scalar(1.0f));
    class_weights.at<double>(0,1) = prior;
    CvSVMParams params;
    params.svm_type = cv::SVM::C_SVC;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000000, 1e-6);
    if (use_rbf) {
        params.kernel_type = cv::SVM::RBF;
        params.gamma = gamma;
    } else {
        params.kernel_type = cv::SVM::LINEAR;
    }
    params.C = C;
    if (auto_train) {
        std::cout << "DOING k-fold CV" << std::endl;
        svm.train_auto(samples, labels, cv::Mat(), cv::Mat(), params, k, grid);
    } else {
        std::cout << "Training SVM" << std::endl;
        svm.train(samples, labels, cv::Mat(), cv::Mat(), params);
    }
    cv::FileStorage fs(output_file.c_str(), cv::FileStorage::WRITE);
    svm.write(*fs, "trees");

    int errs = 0;
    for (size_t i = 0; i < samples.rows; i++) {
        std::cout << svm.predict(samples.row(i)) << " " << labels.at<float>(i,0) << std::endl;
        if (svm.predict(samples.row(i)) * labels.at<float>(i,0) < 0) {
            errs++;
        }
    }
    std::cout << "Training Error: " << float (errs) / labels.rows << std::endl;
    return 0;
}

