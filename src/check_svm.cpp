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

#include <getopt.h>

class MySVM : public cv::SVM
{
public:
    CvSVMDecisionFunc* get_decision_func() { return decision_func; }
    MySVM() {}
    ~MySVM() {}
};


int main(int argc, const char *argv[])
{
    if (argc <= 1) return -1;
    MySVM svm;
    cv::FileStorage fs("model_svm_cc.yml", cv::FileStorage::READ);
    svm.read(*fs, *fs["trees"]);

    cv::TrainData data;
    data.read_csv(argv[1]);
    data.set_response_idx(0);
    data.set_delimiter(',');

    cv::Mat values = cv::Mat(data.get_values());
    cv::Mat samples = values.colRange(1, values.cols);
    cv::Mat labels = values.colRange(0, 1);

    static cv::Mat w(1, samples.cols, CV_32FC1, cv::Scalar(0.0f));
    static float b = 0;

    int c = svm.get_support_vector_count();
    for (unsigned int i = 0; i < c; ++i) {
        float alpha = svm.get_decision_func()->alpha[i];
        int idx = svm.get_decision_func()->sv_index[i];
        const float *val = svm.get_support_vector(idx);

        for (unsigned int j = 0; j < w.cols; ++j) {
            w.at<float>(0,j) += (val[j] * alpha);
        }
    }
    //cv::normalize(w, w);

    b = -svm.get_decision_func()->rho;

    int errs = 0;
    int fps = 0;
    int fns = 0;
    int pos_total = 0;
    int neg_total = 0;
    for (int i = 0; i < samples.rows; ++i) {
        float result = (samples.row(i).dot(w) + b) > 0 ? -1 : 1;
        //float result = svm.predict(samples.row(i));
        if (labels.at<float>(i,0) > 0) ++pos_total;
        else ++neg_total;
        if (result != labels.at<float>(i,0)) {
            ++errs;
            if (labels.at<float>(i,0) > 0)  {
                fns++;
            } else {
                fps++;
            }
        }
    }
    std::cout << "Error: " << float(errs) / samples.rows << " false positives: " << float(fps) / neg_total
                << " false negatives: " << float(fns) / pos_total << std::endl;

    std::cout << w << std::endl;
    std::cout << b << std::endl;
}
