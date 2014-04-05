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
#include <text_detector/UnionFind.h>

namespace TextDetector {

UnionFind::UnionFind(int n_nodes)
{
    _parents.resize(n_nodes, -1);
    _ranks.resize(n_nodes, 0);
}
UnionFind::~UnionFind(){}

int UnionFind::find_set(int id) 
{
    if (_parents[id] == -1)
        return id;
    _parents[id] = find_set(_parents[id]);
    return _parents[id];
}

void UnionFind::union_set(int i, int j)
{
    int p_i = find_set(i);
    int p_j = find_set(j);
    if (p_i != p_j) {
        if (_ranks[p_i] < _ranks[p_j]) {
            _parents[p_i] = p_j;
        } else {
            if (_ranks[p_i] == _ranks[p_j]) {
                _ranks[p_j] += 1;
            }
            _parents[p_j] = p_i;
        }
    }
}

}
