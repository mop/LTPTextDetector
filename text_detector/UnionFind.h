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
#ifndef UNIONFIND_H

#define UNIONFIND_H

#include <vector>

namespace TextDetector {

/**
 * Simple implementation of Union Find
 */
class UnionFind 
{
public:
    UnionFind(int n_nodes);
    ~UnionFind();

    int find_set(int id);
    void union_set(int i, int j);

private:
    std::vector<int> _parents;
    std::vector<int> _ranks;
};

}

#endif /* end of include guard: UNIONFIND_H */
