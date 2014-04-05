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
#include <opencv2/highgui/highgui.hpp>
#include <detector/Adaboost.h>
#include <text_detector/ModelManager.h>
#include <text_detector/ConfigurationManager.h>
#include <text_detector/AdaboostClassifier.h>

#include <boost/filesystem.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include <text_detector/MserDetector.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, const char *argv[])
{
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "print this help message")
            ("config,c", po::value<std::string>()->required(), "path to config file")
            ("model,m", po::value<std::string>()->required(), "path to model file")
            ("input,i", po::value<std::string>()->required(), "path to image file")
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

        srand(config->get_random_seed());

        // creates the models
        TextDetector::ModelManager model_manager(config);

        if (config->verbose())
            std::cout << "Read inputs" << std::endl;

        std::string input(vm["input"].as<std::string>());
        if (!fs::exists(input)) {
            std::cerr << "Input image not available" << std::endl;
            return 1;
        }

        TextDetector::MserDetector detector(config);

        TextDetector::AdaboostClassifier clf(vm["model"].as<std::string>());

        cv::Mat image = cv::imread(input);
        cv::Mat mask;

        clf.detect(image, mask);
        cv::Mat result_image;
        std::vector<cv::Rect> words = detector(image, result_image, mask > (255 * threshold));

        for (size_t i = 0; i < words.size(); i++) {
            cv::Rect r = words[i];
            cv::rectangle(result_image, r.tl(), r.br(), cv::Scalar(0, 0, 255), 4);
            std::cout << r.x << "," << r.y << "," << r.width << "," << r.height << std::endl;
        }
        cv::imshow("MASK", mask); 
        cv::imshow("RESULT", result_image); cv::waitKey(0);


    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
