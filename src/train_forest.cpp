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
 * This is a helper program for training random forests.
 */
int main(int argc, char *argv[])
{
    int random_seed = 42;

    int c;
    std::string train_file;
    std::string output_file;
    int ntrees = 100;
    int ndepth = 5;
    int nvars = 10;
    int nmin_samples = 20;
    float prior = 1.0f;
    bool calc_var_importance = false;
    bool log_errs = false;
    bool surrogates = false;

    while ((c = getopt(argc, argv, "r:i:o:n:d:v:p:m:c:lsh")) != -1) {
        switch (c) {
            case 'i':
                train_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'n':
                std::stringstream(optarg) >> ntrees;
                break;
            case 'm':
                std::stringstream(optarg) >> nmin_samples;
                break;
            case 'd':
                std::stringstream(optarg) >> ndepth;
                break;
            case 'v':
                std::stringstream(optarg) >> nvars;
                break;
            case 'p':
                std::stringstream(optarg) >> prior;
                break;
            case 'c':
                std::stringstream(optarg) >> calc_var_importance;
                break;
            case 'l':
                log_errs = true;
                break;
            case 's':
                surrogates = true;
                break;
            case 'r':
                std::stringstream(optarg) >> random_seed;
                break;
            case 'h':
            default:
                std::cerr << "Usage: train_forest OPTIONS" << std::endl 
                    << "\t -i <train.csv>" << std::endl 
                    << "\t -o <out.txt>" << std::endl 
                    << "\t -r <random seed>" << std::endl 
                    << "\t -n <number of trees>" << std::endl 
                    << "\t -m <minimum number of samples in a leaf>" << std::endl 
                    << "\t -v <number of active variables>" << std::endl 
                    << "\t -p <prior/weight of text>" << std::endl 
                    << "\t -c <calc variable importance>" << std::endl 
                    << "\t -l <log errors>" << std::endl 
                    << "\t -s <use surrogates>" << std::endl 
                    << "\t -d <depth of trees>" << std::endl;
                return 1;
        }
    }
    srand(random_seed);
    if (ndepth < 1) { 
        std::cerr << "Error - depth of trees must be at least 1" << std::endl;
        return 1;
    }
    if (ntrees < 1) {
        std::cerr << "Error - need at least 1 tree" << std::endl;
        return 1;
    }
    if (nvars < 1) {
        std::cerr << "Error - need at least 1 variable" << std::endl;
        return 1;
    }
    if (nmin_samples < 1) {
        std::cerr << "Error - need at least 1 sample" << std::endl;
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
    std::cout << ntrees << " " << ndepth << " " << nvars << std::endl;

    cv::TrainData data;
    data.read_csv(train_file.c_str());
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat values = cv::Mat(data.get_values());
    cv::Mat samples = values.colRange(1, values.cols);
    cv::Mat labels = values.colRange(0, 1);

    std::cout << labels.rows << " " << labels.cols << std::endl;
    std::cout << samples.rows << " " << samples.cols << std::endl;


    cv::RandomTrees trees;
    float priors[] = { 1.0f, prior };
    cv::Mat vartype(samples.cols+1, 1, CV_8U);
    vartype.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    //vartype.setTo(cv::Scalar(CV_VAR_CATEGORICAL));
    vartype.at<uchar>(samples.cols, 0) = CV_VAR_CATEGORICAL;
    trees.train(samples, CV_ROW_SAMPLE, labels, cv::Mat(), cv::Mat(), vartype, cv::Mat(), CvRTParams(
        ndepth,                // max depth
        nmin_samples,          // min sample count
        0,                     // regression accuracy
        surrogates,            // use surrogates
        15,                    // max categories?!
        priors,                // priors
        calc_var_importance,   // calc var importace
        nvars,                 // nactiv vars?!
        ntrees,                // max. no of trees
        0.01f,                 // forest accuracy,
        CV_TERMCRIT_ITER));
    cv::FileStorage fs(output_file.c_str(), cv::FileStorage::WRITE);
    trees.write(*fs, "trees");

    int false_positives = 0;
    int false_negatives = 0;
    int num_positives = 0;
    int num_negatives = 0;
    for (int i = 0; i < samples.rows; ++i) {
        float prob = trees.predict_prob(samples.row(i));
        float lbl = trees.predict(samples.row(i));
        if (prob > 0.5 && lbl != 1) { 
            std::cout << prob << " " << lbl << std::endl;
        }
        if (labels.at<float>(i,0) == -1) {
            ++num_negatives;
        } else {
            ++num_positives;
        }
        if (prob > 0.5 && labels.at<float>(i,0) == -1) {
            ++false_positives;
            if (log_errs) {
                std::cout << i << std::endl;
            }
        } else if (prob <= 0.5 && labels.at<float>(i,0) == 1) {
            ++false_negatives;
            if (log_errs) {
                std::cout << i << std::endl;
            }
        }
    }

    std::cout << "Training finished with false positives: " << (float(false_positives) / float(num_negatives)) 
              << " and false negatives: "  << (float(false_negatives) / float(num_positives)) << std::endl;
    std::cout << "Num negatives: " << num_negatives << " Num positives: " << num_positives << " false positives: " << false_positives << " false negatives: " << false_negatives << std::endl;
    
    if (calc_var_importance) {
        cv::Mat importance = trees.get_var_importance();
        std::cout << importance << std::endl;
    }
    
    return 0;
}
