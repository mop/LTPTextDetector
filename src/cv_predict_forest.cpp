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
    std::string train_file, output_file;
    int random_seed = 42;
    int ntrees = 10;
    int depth = 10;
    int nvars = 10;
    int nfolds = 5;
    while ((c = getopt(argc, argv, "r:i:n:d:v:f:o:h")) != -1) {
        switch (c) {
            case 'r':
                std::stringstream(optarg) >> random_seed;
                break;
            case 'f':
                std::stringstream(optarg) >> nfolds;
                break;
            case 'i':
                train_file = optarg;
                break;
            case 'n':
                std::stringstream(optarg) >> ntrees;
                break;
            case 'd':
                std::stringstream(optarg) >> depth;
                break;
            case 'v':
                std::stringstream(optarg) >> nvars;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'h':
            default:
                std::cerr << "Usage: train_forest OPTIONS" << std::endl 
                    << "\t -i <train.csv>" << std::endl 
                    << "\t -f <folds>" << std::endl 
                    << "\t -r <random_seed>" << std::endl 
                    << "\t -n <number of trees>" << std::endl 
                    << "\t -v <number of active>" << std::endl 
                    << "\t -o <out file>" << std::endl 
                    << "\t -d <depth of trees start>" << std::endl;
                return 1;
        }
    }
    srand(random_seed);
    if (depth < 1) {
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
    if (train_file == "" || output_file == "") {
        std::cerr << "Error - need a training/output file" << std::endl;
        return 1;
    }
    std::cout << train_file << std::endl;
    std::cout << nfolds << std::endl;
    std::cout << ntrees << std::endl
              << nvars << std::endl
              << depth << std::endl;

    cv::TrainData data;
    data.read_csv(train_file.c_str());
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat values = cv::Mat(data.get_values());
    cv::Mat result_labels(values.rows, 1.0, CV_32FC1, cv::Scalar(0.0f));
    std::vector<int> indices(values.rows);
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

    // randomly permute them
    for (int i = 0; i < values.rows; i++) {
        int j = rand() % values.rows;
        if (i == j) continue;
        cv::Mat tmp;
        values.row(i).copyTo(tmp);

        cv::Mat row_j = values.row(j);
        cv::Mat row_i = values.row(i);

        row_j.copyTo(row_i);
        tmp.copyTo(row_j);

        std::swap(indices[i], indices[j]);
    }

    cv::Mat samples = values.colRange(1, values.cols);
    cv::Mat labels = values.colRange(0, 1);

    std::cout << labels.rows << " " << labels.cols << std::endl;
    std::cout << samples.rows << " " << samples.cols << std::endl;
    std::cout << samples.at<float>(0,0) << std::endl;

    int chunk_size = labels.rows / nfolds;
    std::cout << "CHUNK SIZE: "  << chunk_size << " " << labels.rows << std::endl;

                float err = 0;
                float fp_err = 0;
                float fn_err = 0;
                float precision = 0;
                float recall = 0;
                for (int chunk = 0; chunk < nfolds; ++chunk) {
                    cv::RandomTrees trees;
                    float priors[] = { 1.0f,1.0f };
                    cv::Mat vartype(samples.cols+1, 1, CV_8U);
                    vartype.setTo(cv::Scalar(CV_VAR_NUMERICAL));
                    vartype.at<uchar>(samples.cols, 0) = CV_VAR_CATEGORICAL;
                    int idx_start = chunk * chunk_size;
                    int idx_end = std::min(labels.rows - 1, idx_start + chunk_size - 1);

                    std::cout << idx_start << " - " << idx_end << std::endl;
                    cv::Mat holdout = samples.rowRange(idx_start, idx_end);
                    cv::Mat holdout_labels = labels.rowRange(idx_start, idx_end);
                    //cv::Mat holdout = cv::Mat(samples, cv::Range(idx_start, idx_end),cv::Range::all());
                    //cv::Mat holdout_labels = cv::Mat(labels, cv::Range(idx_start, idx_end),cv::Range::all());

                    cv::Mat train_set = cv::Mat(samples.rows - holdout.rows, samples.cols, CV_32F);
                    cv::Mat train_labels = cv::Mat(samples.rows - holdout.rows, labels.cols, CV_32F);
                    int idx = 0;
                    for (int r = 0; r < labels.rows; ++r) {
                        if (r < idx_start || r >= idx_end) {
                            for (int c = 0; c < train_set.cols; ++c) {
                                 train_set.at<float>(idx, c) = samples.at<float>(r, c);
                            }
                            train_labels.at<float>(idx, 0) = labels.at<float>(r,0);

                            ++idx;
                        }
                    }
                    
                    std::cout << "holdout rows: " << holdout.rows << std::endl;
                    std::cout << "TRAINING " << idx << " " << train_labels.rows << std::endl;
                    trees.train(train_set, CV_ROW_SAMPLE, train_labels, cv::Mat(), cv::Mat(), vartype, cv::Mat(), CvRTParams(
                        depth,  // max depth
                        5,      // min sample count
                        0,      // rgression accuracy
                        false,  // use surrogates
                        15,     // max categories?!
                        NULL, // priors
                        false,  // calc var importace
                        nvars,  // nactiv vars?!
                        ntrees, // max. no of trees
                        0.01f,  // forest accuracy,
                        CV_TERMCRIT_ITER | CV_TERMCRIT_EPS));

                    int nmissclass_train = 0;
                    for (int i = 0; i < train_set.rows; ++i) {
                        float res = trees.predict_prob(train_set.row(i));
                        res = res < 0.5f ? -1.0f : 1.0f;
                        if (res != train_labels.at<float>(i, 0)) {
                            ++nmissclass_train;
                        }
                    }
                    std::cout << "Train Error: " << float (nmissclass_train) / float(train_labels.rows) << std::endl;
                    int nmissclass = 0; 
                    int nfalse_pos = 0; 
                    int nfalse_neg = 0; 
                    int real_neg = 0;
                    int real_pos = 0;
                    int true_pos = 0;
                    int true_neg = 0;
                    for (int i = 0; i < holdout.rows; ++i) {
                        float res = trees.predict_prob(holdout.row(i));
                        // store the result
                        result_labels.at<float>(indices[i + idx_start], 0) = res;
                        // text
                        res = res < 0.5f ? -1.0f : 1.0f;
                        if (holdout_labels.at<float>(i,0) > 0) {
                            real_pos++;
                            if (res > 0) {
                                true_pos++;
                            }
                        } else {
                            real_neg++;
                            if (res <= 0) {
                                true_neg++;
                            }
                        }
                        if (res != holdout_labels.at<float>(i, 0)) {
                            ++nmissclass;
                            if (res == 1 && holdout_labels.at<float>(i,0) == -1) {
                                ++nfalse_pos;
                            } else {
                                ++nfalse_neg;
                            }
                        }
                    }

                    err += float (nmissclass) / float (holdout_labels.rows);
                    fp_err += float (nfalse_pos) / float (nfalse_pos + real_neg);
                    fn_err += float (nfalse_neg) / float (nfalse_neg + real_neg);
                    precision += float (true_pos) / (nfalse_pos + true_pos);
                    recall += float (true_pos) / (nfalse_neg + true_pos);
                    std::cout << "err: " << float (nmissclass) / float (holdout_labels.rows) << std::endl;
                    std::cout << "false pos: " << float (nfalse_pos) / float (nfalse_pos + real_neg) << std::endl;
                    std::cout << "false neg: " << float (nfalse_neg) / float (nfalse_neg + real_neg) << std::endl;
                }

                err = err / nfolds;
                fp_err = fp_err / nfolds;
                fn_err = fn_err / nfolds;
                precision = precision / nfolds;
                recall = recall / nfolds;
                std::cout << "CV Error for ntrees: " << ntrees << " nvars: " 
                          << nvars << " depth: " << depth << " error: " << err << " (fp: " << fp_err << " fn: " << fn_err << " p: " << precision << " r: " << recall << ")" << std::endl;


                std::ofstream ofs(output_file.c_str());
                for (int i = 0; i < result_labels.rows; i++) {
                    ofs << result_labels.at<float>(i, 0) << std::endl;
                }
                ofs.close();
    return 0;


}

