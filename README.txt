LTPTextDetector
===========================================

This repository contains the open source release of the research publication

End-to-End Text Recognition using Local Ternary Patterns, MSER and Deep Convolutional Neural Networks
Michael Opitz, Markus Diem, Markus Diem, Florian Kleber, Stefan Fiel and Robert Sablatnig 
presented at DAS 2014.

Requirements
===========================================

* Linux (untested under Windows, OS X)
* Boost
* OpenCV 2.4
* CMake
* Eigen 3
* python 2.7
* LTPTextDetectorTraining (available on github)
* Time and patience to get things running

How to compile the code?
===========================================

Before compiling the code, grab the LTPTextDetectorTraining project on GitHub
and extract/symlink it in the detector subdirectory.

Then the project can be compiled by $ cmake . && make

Since GitHub does not allow big files in their repositories, pre-trained models
have to be downloaded at http://bit.ly/1ehC3ZT and unzipped in the models/ directory

How to run a demo?
===========================================

Just run 

    $ ./bin/demo -c config_11.yml -model models/model_boost.txt -i <image>

from the root-directory of the project. The model files
must be downloaded and extracted in the models directory, as explained in the previous step.


How to reproduce the results?
===========================================

To reproduce the results, download the archive of datasets from http://bit.ly/1gxI9Fx
and unzip it in the parent directory of the project.

Then run  

    $ ./bin/classify -t ../test_icdar_2011  -r ../result_test -m models/model_boost.txt

to create the response maps and 
    
    $ ./bin/create_boxes -c config_11.yml

to create the bounding boxes.
To convert the output to the ICDAR evalution format, run

    $ python2 ./scripts/to_xml.py result_test/ > eval11.xml
    $ evaldetection eval11.xml datasets/test-textloc-gt/test-gt-textloc-wolf.xml > results.xml
    $ readdeteval results.xml 

Which shoult print: 

    Included 255 images with non-zero groundtruth
    Included 0 images with zero groundtruth
    Skipped 0 images with zero groundtruth.
    Total-Number-Of-Processed-Images: 255
    100% of the images contain objects.
    Generality: 4.66275
    Inverse-Generality: 0.214466
    <evaluation noImages="255">
      <icdar2003 r="0.700094" p="0.81904" hmean="0.75491" noGT="1189" noD="1026"/>
      <score r="0.715559" p="0.844055" hmean="0.774514" noGT="1189" noD="1026"/>
    </evaluation>

How to retrain the models?
===========================================

Training scripts are in the scripts/ subdirectory. To retrain the models unzip the datasets 
in the parent directory of the repository. 
To retrain everything from scratch run

    $ ./script/train_all.sh

What about the Recognizer?
===========================================

Comming soon...
