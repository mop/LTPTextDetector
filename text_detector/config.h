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
#ifndef MY_CONFIG_H

#define MY_CONFIG_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <dlib/svm_threaded.h>
#pragma GCC diagnostic pop


typedef dlib::matrix<double,2,1> node_vector_type;
typedef dlib::matrix<double,4,1> edge_vector_type;
typedef dlib::graph<node_vector_type, edge_vector_type>::kernel_1a_c graph_type;
typedef dlib::matrix<double,0,1> vector_type;

#define N_NEIGHBORS 5

#define CC_HOG_CHANS 8
//#define N_UNARY_FEATURES (11+4 + CC_HOG_CHANS * 3)
#define N_UNARY_FEATURES (15)

#define WINDOW_HEIGHT 12
#define WINDOW_WIDTH 24

#endif /* end of include guard: CONFIG_H */
