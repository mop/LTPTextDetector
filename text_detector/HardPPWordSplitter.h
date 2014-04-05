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

#ifndef HARDPPWORDSPLITTER_H

#define HARDPPWORDSPLITTER_H

#include "WordSplitter.h"

namespace TextDetector {

class HardPPWordSplitter : public WordSplitter
{
public:
    HardPPWordSplitter(bool allow_single_letters=false, bool verbose=false);
    virtual ~HardPPWordSplitter() {}

    virtual std::vector<cv::Rect> split(const CCGroup &grp);
private:
    bool _allow_single_letters;
    bool _verbose;
};

}

#endif /* end of include guard: HARDPPWORDSPLITTER_H */
