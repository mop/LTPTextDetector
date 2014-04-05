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

#ifndef MSERTREE_H
#define MSERTREE_H

#include "HierarchicalMSER.h"

#include <memory>

namespace TextDetector {

/**
 * This class represents a node of a mser tree
 */
class MserNode : public std::enable_shared_from_this<MserNode>
{
public:
    MserNode() {}
    ~MserNode() {}

    //! Draws the tree into the given image
    void draw_tree(cv::Mat &img);

    //! Writes the tree into the graph/node<uid>.png directory
    void write_tree(const cv::Size &dim);

    //! Prints the node as graphviz thing
    void print(int parent);
    //! Prints the node as graphviz thing
    void print_node(int parent);
    //! Dumps the tree into dirname/node<uid>.png and dirname/node<uid>_contour.csv
    void dump_tree(const std::string &dirname, const cv::Size &dim);
    //! Dumps the tree as csv format as format uid,parent,prob,x,y,w,h
    void dump_csv(std::ofstream &ofs, int parent=0);
    //! Dumps the tree as graphviz format
    void dump_gv(std::ofstream &ofs, int parent=0);
    //! Draws the node into the given image
    void draw_node(cv::Mat img, const cv::Vec3b &color = cv::Vec3b(0,255,0));
    //! Finds the node with the given uid
    std::shared_ptr<MserNode> find(int uid) const;
    //! Finds the node which overlaps with the given coordinate
    int find_by_coordinates(int x, int y, float min_area = 0, float max_area = FLT_MAX) const;
    //! Finds all children fully contained in the given bounding rect
    std::vector<std::shared_ptr<MserNode> > find_by_bounding_rect(const cv::Rect &r) const;
    //! Matches the given contour with the own/its children
    int match_contour(const cv::Mat &contour, float thresh = 0.9f) const;

    //! Linearized the tree
    std::shared_ptr<MserNode> linearize();
    //! Accumulates the tree
    std::vector<std::shared_ptr<MserNode> > accumulate();

    //! Stores the pointer to the children
    std::vector<std::shared_ptr<MserNode> > children;
    //! Stores the contour
    std::vector<cv::Point> contour;
    //! Stores the probability
    double prob;
    //! Stores the bounding rect
    cv::Rect rect;
    //! Stores the uid of the node
    int uid;
    //! Stores the index of the node into the contour array
    int idx;
};

/**
 *  This class represents an tree of mser nodes
 */
class MserTree 
{
public:
    MserTree(
        const std::vector<std::vector<cv::Point> > &contours,
        const std::vector<double> &probs,
        const std::vector<cv::Vec4i> &hierarchy, 
        const cv::Mat &mask = cv::Mat(),
        float thresh = 0.5f * 255);

    void print();
    void print_child(int i);

    void dump_child(const std::string &dirname, int i);

    void print_collapsed(int i);
    void linearize();
    void accumulate();
    void prune(const cv::Mat &mask, float thresh = 0.5f, bool exact = false);

    std::vector<std::vector<cv::Point> > get_accumulated_contours() const;
    std::vector<int> get_accumulated_indices() const;

    ~MserTree() { }

    std::shared_ptr<MserNode> find(int uid);
    //! Finds the first children overlapping w/ the given coordinates, 
    //! having an area between min_area and max_area
    int find_by_coordinates(int x, int y, float min_area=0, float max_area = FLT_MAX) const;
    //! Finds all children which are enclosed by the given bounding rect
    std::vector<std::shared_ptr<MserNode> > find_by_bounding_rect(const cv::Rect &r) const;
    //! Matches the given contour with the own/its children
    int match_contour(const cv::Mat &contour, float thresh = 0.9f) const;

public:
    std::shared_ptr<MserNode> _root;
    std::vector<std::vector<std::shared_ptr<MserNode> > > _accumulated_nodes;
    std::vector<std::vector<int> > _accumulated_indices;
};

}


#endif /* end of include guard: MSERTREE_H */
