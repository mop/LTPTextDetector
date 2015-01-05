project(text_detection)
cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
find_package(Qt4 REQUIRED)
find_package(OpenMP REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED program_options filesystem system serialization timer)

option(DEBUG "Debugging Mode" OFF)
option(EIGEN3_INCLUDE_DIR "Eigen3 Include Directory" "/usr/include/eigen3")


get_filename_component(ext_dir "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/" ABSOLUTE)

## Compiler flags
if (DEBUG)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "-I${ext_dir}/libsvm-3.16/ -g -pg -std=c++11")        ## Debug
    endif()
else()
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "-I${ext_dir}/libsvm-3.16/ -O3 -march=native -fopenmp -std=c++11 -D__INTRIN__ENABLED__=1 -Wpedantic -funroll-loops -ftree-vectorize")        ## Production
    endif()
endif()

IF (${EIGEN3_INCLUDE_DIR} MATCHES "OFF")
    SET(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
ENDIF(${EIGEN3_INCLUDE_DIR} MATCHES "OFF")

include_directories("." "${ext_dir}/dlib-18.0/" "detector/" ${EIGEN3_INCLUDE_DIR})
link_directories("${ext_dir}/dlib-18.0/dlib/")
link_directories("${ext_dir}/../detector")
link_directories("${ext_dir}/../.")
link_directories("${CMAKE_BINARY_DIR}")
link_directories("${CMAKE_BINARY_DIR}/detector")
link_directories("${CMAKE_BINARY_DIR}/3rdparty/dlib-18.0/dlib/")

set(Dlib_LIBRARIES "-ldlib")

add_subdirectory("detector")
add_subdirectory("3rdparty/dlib-18.0/dlib/")

QT4_WRAP_CPP(MOCSrcs text_detector/LabelWidget.h)
include(${QT_USE_FILE})

file(GLOB_RECURSE library_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp)
list(REMOVE_ITEM library_src 
    src/LabelWidget.cpp 
    src/check_svm.cpp
    src/classify.cpp
    src/create_boxes.cpp
    src/cv_forest.cpp
    src/cv_predict_forest.cpp
    src/extract_adjacent_neighbors.cpp
    src/extract_cc_features.cpp
    src/extract_dists.cpp
    src/extract_hog_features.cpp
    src/extract_mser_cc.cpp
    src/extract_train_set.cpp
    src/predict_crf2.cpp
    src/predict_forest.cpp
    src/train_adaboost.cpp
    src/train_crf2.cpp
    src/train_forest.cpp
    src/demo.cpp
    src/train_svm.cpp
)
list(APPEND library_src ${ext_dir}/libsvm-3.16/svm.cpp)

message("Using eigen3: ${EIGEN3_INCLUDE_DIR}")

add_library(text_detect SHARED ${library_src})
set_target_properties(text_detect PROPERTIES OUTPUT_NAME text_detect)
target_link_libraries(text_detect ${OpenCV_LIBS} ${BOOST_LIBRARIES} ${Dlib_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS})

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory bin)

# Labeling tool
add_executable(bin/extract_mser_cc src/extract_mser_cc.cpp src/LabelWidget.cpp ${MOCSrcs})

# TO-Polish:
add_executable(bin/classify src/classify.cpp)
add_executable(bin/extract_train_set src/extract_train_set.cpp src/LTPComputer.cpp)

# most important files
add_executable(bin/demo src/demo.cpp)
add_executable(bin/create_boxes src/create_boxes.cpp)
add_executable(bin/extract_cc_features src/extract_cc_features.cpp)
add_executable(bin/extract_hog_features src/extract_hog_features.cpp)
add_executable(bin/extract_adjacent_neighbors src/extract_adjacent_neighbors.cpp)
add_executable(bin/extract_dists src/extract_dists.cpp)

# ML-training helper programs 
add_executable(bin/train_crf2 src/train_crf2.cpp)
add_executable(bin/predict_crf2 src/predict_crf2.cpp)           
add_executable(bin/train_forest src/train_forest.cpp)
add_executable(bin/predict_forest src/predict_forest.cpp)
add_executable(bin/cv_forest src/cv_forest.cpp)
add_executable(bin/cv_predict_forest src/cv_predict_forest.cpp)
add_executable(bin/check_svm src/check_svm.cpp)

target_link_libraries(bin/demo ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Dlib_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS} -lgomp -ljpeg -lpng -lX11 -ltext_detect -ladaboost)
target_link_libraries(bin/cv_forest ${OpenCV_LIBS})
target_link_libraries(bin/cv_predict_forest ${OpenCV_LIBS})
target_link_libraries(bin/train_forest ${OpenCV_LIBS})
target_link_libraries(bin/predict_forest ${OpenCV_LIBS})
target_link_libraries(bin/train_crf2 ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Dlib_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS} -lgomp -ljpeg -lpng -lX11)
target_link_libraries(bin/predict_crf2 ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Dlib_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS} -lgomp -ljpeg -lpng -lX11)
target_link_libraries(bin/extract_train_set ${OpenCV_LIBS})
target_link_libraries(bin/check_svm ${OpenCV_LIBS})
target_link_libraries(bin/classify ${OpenCV_LIBS} ${Boost_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS} -lgomp -ltext_detect -ladaboost)
target_link_libraries(bin/extract_mser_cc ${OpenCV_LIBS} ${Boost_LIBRARIES} ${QT_LIBRARIES} -ltext_detect -ladaboost)
target_link_libraries(bin/extract_cc_features ${OpenCV_LIBS} ${Boost_LIBRARIES} -ltext_detect -ladaboost)
target_link_libraries(bin/extract_hog_features ${OpenCV_LIBS} ${Boost_LIBRARIES} -ltext_detect -ladaboost)
target_link_libraries(bin/create_boxes ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Dlib_LIBRARIES} -lgomp -ljpeg -lpng -lX11 ${OpenMP_EXE_LINKER_FLAGS} -ltext_detect -ladaboost)
target_link_libraries(bin/extract_adjacent_neighbors ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Dlib_LIBRARIES} -lgomp -ljpeg -lpng -lX11 ${OpenMP_EXE_LINKER_FLAGS} -ltext_detect -ladaboost)
target_link_libraries(bin/extract_dists ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Dlib_LIBRARIES} -lgomp -ljpeg -lpng -lX11 ${OpenMP_EXE_LINKER_FLAGS} -ltext_detect -ladaboost)

add_dependencies(bin/extract_mser_cc text_detect dlib adaboost)
add_dependencies(bin/create_boxes text_detect dlib adaboost)
add_dependencies(bin/extract_cc_features text_detect dlib)
add_dependencies(bin/extract_adjacent_neighbors text_detect dlib)
add_dependencies(bin/extract_dists text_detect dlib)
add_dependencies(bin/classify text_detect adaboost dlib)
add_dependencies(bin/demo text_detect adaboost dlib)
