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
#ifndef WORDSPLITTER_H
#define WORDSPLITTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include "CCGroup.h"

namespace TextDetector {

/**
 * This class is responsible for splitting a CCGroup into words.
 */
class WordSplitter 
{
public:
    WordSplitter() {}
    virtual ~WordSplitter() {}

    /**
     *  Splits the given group into a list of words represented as rectangles.
     */
    virtual std::vector<cv::Rect> split(const CCGroup &grp) = 0;
private:
};

}


#endif /* end of include guard: WORDSPLITTER_H */
