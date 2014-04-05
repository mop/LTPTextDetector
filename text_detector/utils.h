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
#ifndef UTILS_H

#define UTILS_H

static std::vector<std::string> split(const std::string &s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> result;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

template <class T>
static std::vector<T> split(const std::string &s, char delim)
{
    std::vector<std::string> v = split(s, delim);
    std::vector<T> result;
    for (int i = 0; i < v.size(); ++i) {
        T val;
        std::istringstream ss(v[i]);
        ss >> val;
        result.push_back(val);
    }
    return result;
}

static std::vector<float> splitf(const std::string &s, char delim)
{
    return split<float>(s, delim);
}

static std::vector<int> spliti(const std::string &s, char delim)
{
    return split<int>(s, delim);
}



#endif /* end of include guard: UTILS_H */
