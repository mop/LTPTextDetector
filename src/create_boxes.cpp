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
#include <text_detector/config.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include <string>
#include <iostream>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <text_detector/HierarchicalMSER.h>
#include <text_detector/HogIntegralImageComputer.h>
#include <text_detector/MserTree.h>
#include <text_detector/LibSVMClassifier.h>
#include <text_detector/CCUtils.h>
#include <text_detector/MserExtractor.h>
#include <text_detector/MserExtractorFast.h>
#include <text_detector/CCGroup.h>
#include <text_detector/UnionFind.h>

#include <text_detector/ConfigurationManager.h>
#include <text_detector/ModelManager.h>
#include <text_detector/ConnectedComponentFilterer.h>
#include <text_detector/CRFLinConnectedComponentFilterer.h>
#include <text_detector/CRFRFConnectedComponentFilterer.h>
#include <text_detector/RFConnectedComponentFilterer.h>
#include <text_detector/SoftPPWordSplitter.h>
#include <text_detector/HardPPWordSplitter.h>
#include <text_detector/CacheManager.h>
#include <text_detector/SimpleWordSplitter.h>
#include <text_detector/BinaryMaskExtractor.h>
#include <text_detector/CNNConnectedComponentClassifier.h>
#include <text_detector/RFConnectedComponentClassifier.h>
#include <text_detector/SVMConnectedComponentClassifier.h>
#include <text_detector/SVMRFConnectedComponentClassifier.h>

#include <text_detector/MserDetector.h>

#include <signal.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
typedef dlib::matrix<double,0,1> vector_type;

void
terminate (int sig)
{
    exit(0);
}

int main(int argc, const char *argv[])
{
    signal(SIGINT, terminate);
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "print this help message")
            ("config,c", po::value<std::string>()->required(), "path to config file")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        std::shared_ptr<TextDetector::ConfigurationManager> config(
        	new TextDetector::ConfigurationManager(
        		vm["config"].as<std::string>()));
        TextDetector::ConfigurationManager::set_instance(config);


        float threshold = config->get_threshold();
        if (config->has_cache()) {
            CacheManager::set_instance(
                std::shared_ptr<CacheManager>((new CacheManager(config->get_cache_directory()))));
        }

        srand(config->get_random_seed());

        // creates the models
        TextDetector::ModelManager model_manager(config);

        if (config->verbose())
            std::cout << "Read inputs" << std::endl;

        if (!fs::is_directory(config->get_input_directory()) ||
            !fs::is_directory(config->get_responses_directory())) {
            std::cerr << "Error: response or input directory does not exist" << std::endl;
            return 1;
        }

        std::vector<fs::path> directories;
        std::copy(fs::directory_iterator(
            config->get_input_directory()),
            fs::directory_iterator(), 
            std::back_inserter(directories));
        std::sort(directories.begin(), directories.end());

        TextDetector::MserDetector detector(config);
        for (auto it = directories.begin(); it != directories.end(); ++it) {
            fs::path p(*it);
            //p = "../train_icdar_2005/332.jpg";
            //p = "../test_icdar_2005/214.jpg";

            if (p.extension() != ".jpg" && p.extension() != ".png") {
                std::cout << "Skipping: " << p << std::endl;
                continue;
            }

            fs::path number = p.stem();

            std::string response_name = number.generic_string() + "_response.png";
            fs::path response = config->get_responses_directory();
            response += "/";
            response += response_name;

            if (!fs::exists(response) && 
                !config->ignore_responses()) {
                std::cout << "Error, " << response
                		  << " does not exist -> skipping!" << std::endl;
                continue;
            }

            std::cout << "Processing: " << p.filename() << " " 
                      << number << " " << response << std::endl;

            if (CacheManager::instance()) {
                CacheManager::instance()->load_image(number.generic_string());
            }
            
            cv::Mat response_image;
            if (fs::exists(response)) {
                response_image = cv::imread(response.generic_string());
                if (response_image.channels() > 1) {
                    cv::cvtColor(response_image, response_image, cv::COLOR_RGB2GRAY);
                }
            }
            cv::Mat train_image = cv::imread(p.generic_string());
            cv::Mat mask;

            if (!response_image.empty()) 
                mask = response_image > (threshold * 255);

            if (config->ignore_responses()) {
                mask = cv::Mat(train_image.rows, train_image.cols, CV_8UC1, cv::Scalar(255));
            }

            cv::Mat gt_mask;
            if (config->include_binary_masks()) {
                std::string mask_path = config->get_input_directory() + number.generic_string() + "_mask.png";
                if (!fs::exists(mask_path)) {
                    std::cout << "Skipping: " << mask_path << " due to missing mask!" << std::endl;
                    continue;
                }
                gt_mask = cv::imread(mask_path);
            }
            cv::Mat result_image;
            std::vector<cv::Rect> words = detector(train_image, result_image, mask, gt_mask);

            fs::path out_name(config->get_responses_directory());
            std::string box_name = number.generic_string() + "_boxes.txt";
            out_name += "/"; out_name += box_name;
            std::ofstream ofs(out_name.generic_string());
//            cv::Mat result_img = show_groups_color(
//                groups,
//                cv::Size(train_image.cols, train_image.rows),
//                all_probs, false);
            for (size_t i = 0; i < words.size(); i++) {
                cv::Rect r = words[i];
                cv::rectangle(result_image, r.tl(), r.br(), cv::Scalar(0, 0, 255), 4);
                ofs << r.x << "," << r.y << "," << r.width << "," << r.height << std::endl;
            }
            ofs.close();

            std::string box_img_name = number.generic_string() + "_boxes.png";
            fs::path out_img_name(config->get_responses_directory());
            out_img_name += "/"; out_img_name += box_img_name;
            cv::imwrite(out_img_name.generic_string(), result_image);

            if (CacheManager::instance())
                CacheManager::instance()->save_image(number.generic_string());
        }
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
