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
#include <text_detector/MserTree.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <list>

#include <text_detector/ConfigurationManager.h>

namespace TextDetector {

MserTree::MserTree(const std::vector<std::vector<cv::Point> > &contours, const std::vector<double> &probs, const std::vector<cv::Vec4i> &hierarchy, const cv::Mat &mask, float threshold)
: _root(new MserNode)
{
    std::vector<std::shared_ptr<MserNode> > nodes(contours.size());
    _root->uid = 0;
    int uid = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        // check the contour
        if (mask.rows > 0) {
            float sum = 0.0f;
            for (size_t j = 0; j < contours[i].size(); j++) {
                sum += mask.at<cv::Vec3b>(contours[i][j].y, contours[i][j].x)[0];
            }
            if (sum / contours[i].size() < threshold) {
                continue;
            }
        }
        if (!nodes[i])
            nodes[i].reset(new MserNode());
        nodes[i]->contour = contours[i];
        nodes[i]->rect = cv::boundingRect(contours[i]);
        nodes[i]->prob = probs[i];
        nodes[i]->uid = ++uid;
        nodes[i]->idx = i;
        // its a root
        if (hierarchy[i][3] == -1) {
            _root->children.push_back(nodes[i]);
        } else { // no root :(
            int parent_idx = hierarchy[i][3];
            if (!nodes[parent_idx]) {
                nodes[parent_idx].reset(new MserNode());
            }
            nodes[parent_idx]->children.push_back(nodes[i]);
        }
    }
}

void MserTree::print()
{
    std::cout << "digraph G {" << std::endl;
    std::cout << "rankdir=LR" << std::endl;
    _root->print(0);
    std::cout << "}" << std::endl;
}

void MserTree::print_child(int i) {
    std::cout << "digraph G {" << std::endl;
    std::cout << "rankdir=LR" << std::endl;
    _root->children[i]->print(0);
    std::cout << "}" << std::endl;
}

void MserTree::print_collapsed(int i) {
    std::cout << "digraph G {" << std::endl;
    std::cout << "rankdir=LR" << std::endl;
    for (int j = 0; j < _accumulated_nodes[i].size(); ++j) {
        _accumulated_nodes[i][j]->print_node(0);
    }
    std::cout << "}" << std::endl;
}

void MserTree::linearize()
{
    _root->linearize();
}

void MserTree::accumulate()
{
    for (int i = 0; i < _root->children.size(); i++) {
        std::vector<std::shared_ptr<MserNode> > accum = _root->children[i]->accumulate();
        if (accum.empty()) {
            accum.push_back(_root->children[i]);
        } else if (!accum[0]) {
            accum[0] = _root->children[i];
        }
        _accumulated_nodes.push_back(accum);
        //_root->accumulate();
    }
}

void MserNode::draw_tree(cv::Mat &img)
{
    for (int i = 0; i < children.size(); i++) {
        children[i]->draw_tree(img);
    }
    std::cout << children.size() << " " << prob << std::endl;
    for (int i = 0; i < contour.size(); i++) {
        img.at<cv::Vec3b>(contour[i].y, contour[i].x) = cv::Vec3b(prob * 255, 128, 128);
    }
    cv::imshow("BLUB", img); cv::waitKey(0);
}

void MserNode::draw_node(cv::Mat img, const cv::Vec3b &color)
{
    if (!contour.empty()) {
        for (size_t i = 0; i < contour.size(); i++) {
            cv::Point pt = contour[i];

            img.at<cv::Vec3b>(pt.y, pt.x) = color;
        }
    }
}

void MserNode::write_tree(const cv::Size &dim)
{
    cv::Mat result(dim.height, dim.width, CV_8UC3, cv::Scalar(0,0,0));
    if (!contour.empty()) {
        std::vector<std::vector<cv::Point> > tmp;
        tmp.push_back(contour);
        for (int i = 0; i < contour.size(); i++) {
            result.at<cv::Vec3b>(contour[i].y, contour[i].x) = cv::Vec3b(prob * 255, 128, 128);
        }
        cv::Rect rct = cv::boundingRect(contour);
        result = result.rowRange(rct.y, rct.y + rct.height).colRange(rct.x, rct.x+rct.width);
    }
    std::stringstream strm;
    strm << "graph/node" << uid << ".png";

    cv::imwrite(strm.str(), result);

    for (int i = 0; i < children.size(); i++) {
        children[i]->write_tree(dim);
    }
}

void MserNode::print(int parent) {
    print_node(parent);
    for (int i = 0; i < children.size(); i++) {
        children[i]->print(uid);
    }
}

void MserNode::print_node(int parent) {
    std::cout << "n" << parent << " -> n" << uid << std::endl;
    std::cout << "n" << uid << " [image=\"graph/node" << uid << ".png\", label=\"" << prob 
        << " / " << double(rect.width) / double(rect.height) << "\"]" << std::endl;
}

std::shared_ptr<MserNode> MserNode::linearize()
{
    for (int i = 0; i < children.size(); i++) {
        children[i] = children[i]->linearize();
    }
    if (children.size() == 1) {
        if (children[0]->prob > prob) {
            // leave my node out -> return children
            return children[0];
        } else {
            // kill off the child -> my children == children children!
            children = children[0]->children;
            return shared_from_this();
        }
    }
    // multiple children or no children -> return this
    return shared_from_this(); // this
}

std::vector<std::shared_ptr<MserNode> > MserNode::accumulate()
{
    if (children.size() > 1) {
        std::vector<std::shared_ptr<MserNode> > all_childs;
        for (int i = 0; i < children.size(); i++) {
            std::vector<std::shared_ptr<MserNode> > childs = children[i]->accumulate();
            if (childs.empty()) {
                std::cout << "never reached" << std::endl;
                childs.push_back(children[i]);
            }
            all_childs.insert(all_childs.end(), childs.begin(), childs.end());
        }
        //all_childs.erase(std::remove_if(all_childs.begin(), all_childs.end(), [] (std::shared_ptr<MserNode> node) -> double {
        //    return node->prob <= 0.0f;
        //}), all_childs.end());

        std::vector<double> probs(all_childs.size(), 0.0);
        std::transform(
                all_childs.begin(), 
                all_childs.end(),
                probs.begin(), [] (std::shared_ptr<MserNode> node) -> double {
            return node->prob;
        });

        double max = std::accumulate(
                probs.begin(), 
                probs.end(), FLT_MIN, [] (double a, double b) {
            return std::max(a,b);
        });

        std::vector<int> outliners;
        outliners.reserve(all_childs.size());

        // if an child is not intersecting with the parent 
        // (overlap smaller than 0.2 * child-area)
        // then it is considered as an outlier and accumulated up
        for (int i = 0; i < all_childs.size(); i++) {
            cv::Rect intersect = all_childs[i]->rect & rect;
            float intersect_area = intersect.width * intersect.height;
            float child_area = all_childs[i]->rect.width * all_childs[i]->rect.height;
            if (intersect_area / child_area < 0.2f) {
                outliners.push_back(i);
            }
        }

        if (probs.size() > 0 && (max > prob)) {
            return all_childs;
        } else {
            std::vector<std::shared_ptr<MserNode> > me;
            me.push_back(shared_from_this());
            for (int i = 0; i < outliners.size(); i++) {
                me.push_back(all_childs[outliners[i]]);
            }
            return me;
        }

    }

    std::vector<std::shared_ptr<MserNode> > me;
    me.push_back(shared_from_this());
    return me;
}

std::vector<std::vector<cv::Point> > MserTree::get_accumulated_contours() const
{
    std::list<std::shared_ptr<MserNode> > all_nodes;
    for (unsigned int i = 0; i < _accumulated_nodes.size(); ++i) {
        for (unsigned int j = 0; j < _accumulated_nodes[i].size(); ++j) {
            all_nodes.push_back(_accumulated_nodes[i][j]);
        }
    }
    std::vector<std::vector<cv::Point> > result;
    result.resize(all_nodes.size());
    std::transform(
            all_nodes.begin(),
            all_nodes.end(),
            result.begin(),
            [] (const std::shared_ptr<MserNode> &node) -> std::vector<cv::Point> {
        return node->contour;
    });
    return result;
}

std::vector<int> MserTree::get_accumulated_indices() const
{
    std::list<std::shared_ptr<MserNode> > all_nodes;
    for (unsigned int i = 0; i < _accumulated_nodes.size(); ++i) {
        for (unsigned int j = 0; j < _accumulated_nodes[i].size(); ++j) {
            all_nodes.push_back(_accumulated_nodes[i][j]);
        }
    }
    std::vector<int> idxs;
    idxs.resize(all_nodes.size());
    std::transform(
            all_nodes.begin(),
            all_nodes.end(),
            idxs.begin(),
            [] (const std::shared_ptr<MserNode> &node) -> int {
        return node->idx;
    });
    return idxs;
}

void MserNode::dump_tree(const std::string &dirname, const cv::Size &dim)
{
    // dump the tree node as .png in the directory
    cv::Mat result(dim.height, dim.width, CV_8UC3, cv::Scalar(0,0,0));
    if (!contour.empty()) {
        for (size_t i = 0; i < contour.size(); i++) {
            cv::Point pt = contour[i];
            result.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255,0,0);
        }
        //std::vector<std::vector<cv::Point> > tmp;
        //tmp.push_back(contour);
        //cv::drawContours(result, tmp, 0, cv::Scalar(255, 255, 255));

        cv::Rect rct = cv::boundingRect(contour);
        result = result.rowRange(rct.y, rct.y + rct.height).colRange(rct.x, rct.x+rct.width);
    }
    std::stringstream strm;
    strm << dirname << "/node" << uid << ".png";

    cv::imwrite(strm.str(), result);

    // dump the contour!
    {
        std::stringstream contour_path;
        contour_path << dirname << "/node" << uid << "_contour.csv";
        std::ofstream ofs(contour_path.str());
        for (unsigned int i = 0; i < contour.size(); i++) {
            ofs << contour[i].x << "," << contour[i].y << std::endl;
        }
    }

    for (int i = 0; i < children.size(); i++) {
        children[i]->dump_tree(dirname, dim);
    }
}

void MserNode::dump_csv(std::ofstream &ofs, int parent)
{
    ofs << uid << "," << parent << "," << prob << "," 
        << rect.x << "," << rect.y << "," 
        << rect.width << "," << rect.height << std::endl;
    for (int i = 0; i < children.size(); i++) {
        children[i]->dump_csv(ofs, uid);
    }
}

void MserNode::dump_gv(std::ofstream &ofs, int parent)
{
    ofs << "n" << parent << " -> n" << uid << std::endl;
    ofs << "n" << uid << " [image=\"node" << uid << ".png\", label=\"" << uid 
        << " / " << prob << " / " 
        << double(rect.width) / double(rect.height) << "\"]" << std::endl;
    for (int i = 0; i < children.size(); i++) {
        children[i]->dump_gv(ofs, uid);
    }
}

void MserTree::dump_child(const std::string &dirname, int i)
{
    std::stringstream strm;
    strm << dirname << "/graph.gv";
    std::ofstream ofs(strm.str());

    ofs << "digraph G {" << std::endl;
    ofs << "rankdir=LR" << std::endl;
    _root->children[i]->dump_gv(ofs);
    ofs << "}" << std::endl;
}

void MserTree::prune(const cv::Mat &mask, float thresh, bool exact)
{
    _root->children.erase(
        std::remove_if(
            _root->children.begin(),
            _root->children.end(), [&mask, thresh, exact](const std::shared_ptr<MserNode> node) -> bool {
                cv::Rect rect = node->rect;

                if (!exact) {
                    cv::Mat img = mask.rowRange(
                        rect.y,
                        rect.y + rect.height
                    ).colRange(
                        rect.x,
                        rect.x + rect.width
                    );

                    return double(cv::sum(img)[0]) / (rect.width * rect.height) < thresh;
                } else {
                    double sum = 0.0;
                    for (size_t i = 0; i < node->contour.size(); i++) {
                        if (mask.at<cv::Vec3b>(node->contour[i].y, node->contour[i].x)[0] > 0) {
                            sum += 255.0;
                        }
                    }

                    return sum / double (node->contour.size()) < thresh;
                }

        }), _root->children.end());
}

std::shared_ptr<MserNode> MserTree::find(int uid)
{
    if (_root->uid == uid)
        return _root;

    return _root->find(uid);
}

std::shared_ptr<MserNode> MserNode::find(int uid) const
{
    for (size_t i = 0; i < children.size(); i++) {
        if (children[i]->uid == uid) {
            return children[i];
        }
        std::shared_ptr<MserNode> n(children[i]->find(uid));
        if (n) 
            return n;
    }
    return std::shared_ptr<MserNode>();
}

std::vector<std::shared_ptr<MserNode> > MserNode::find_by_bounding_rect(const cv::Rect &br) const
{
    std::vector<std::shared_ptr<MserNode> > result;
    for (std::shared_ptr<MserNode> n : children) {
        cv::Rect intersect = n->rect & br;
        if (intersect == n->rect) {
            result.push_back(n);
        }
        std::vector<std::shared_ptr<MserNode> > child_result = n->find_by_bounding_rect(br);
        result.insert(result.end(), child_result.begin(), child_result.end());
    }
    return result;
}

std::vector<std::shared_ptr<MserNode> > MserTree::find_by_bounding_rect(const cv::Rect &br) const
{
    return _root->find_by_bounding_rect(br);
}

int MserNode::find_by_coordinates(int x, int y, float min_area, float max_area) const
{
    if (rect.x < x && (rect.x + rect.width) > x &&
        rect.y < y && (rect.y + rect.height) > y && contour.size() >= min_area && contour.size() <= max_area)
    {
        for (size_t i = 0; i < contour.size(); i++) {
            if (contour[i].x == x && contour[i].y == y)
                return uid;
        }
    }

    for (size_t i = 0; i < children.size(); i++) {
        int id = children[i]->find_by_coordinates(x, y, min_area, max_area);
        if (id >= 0) 
            return id;
    }
    return -1;
}

int MserTree::find_by_coordinates(int x, int y, float min_area, float max_area) const
{
    return _root->find_by_coordinates(x, y, min_area, max_area);
}

static cv::Mat to_mat(const std::vector<cv::Point> &contour, const cv::Size &size)
{
    cv::Mat result(size.height, size.width, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < contour.size(); i++) {
        result.at<unsigned char>(contour[i].y, contour[i].x) = 255;
    }
    return result;
}

int MserNode::match_contour(const cv::Mat &other_contour, float thresh) const
{
    cv::Mat my_contour = to_mat(contour, cv::Size(other_contour.cols, other_contour.rows));
    cv::Mat result = my_contour & other_contour;
    double area = std::max(cv::sum(other_contour)[0], (double)contour.size());
    if (cv::sum(result)[0] / area > thresh) {
        return uid;
    }

    for (size_t i = 0; i < children.size(); i++) {
        int res = children[i]->match_contour(other_contour, thresh);
        if (res >= 0) return res;
    }
    return -1;
}

int MserTree::match_contour(const cv::Mat &other_contour, float thresh) const
{
    return _root->match_contour(other_contour, thresh);
}

}
