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
#include <text_detector/CCGroup.h>

#include <opencv2/highgui/highgui.hpp>

namespace TextDetector {

CC::CC(int idx, int feature_idx, const std::vector<cv::Point> &img)
: id(idx), feature_id(feature_idx), pixels(img)
{
    int x_min = INT_MAX;
    int x_max = INT_MIN;
    int y_min = INT_MAX;
    int y_max = INT_MIN;

    int sum_x = 0;
    int sum_y = 0;
    for (int i = 0; i < img.size(); i++) {
        x_min = std::min(x_min, img[i].x);
        x_max = std::max(x_max, img[i].x);
        y_min = std::min(y_min, img[i].y);
        y_max = std::max(y_max, img[i].y);

        sum_x += img[i].x;
        sum_y += img[i].y;
    }

    rect = cv::Rect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1);
    centroid = cv::Vec2f(float(sum_x) / img.size(), float(sum_y) / img.size());
}

bool CC::can_link(const CC &rhs) const
{
    float dist_thresh = 2 * std::min(std::max(rect.height, rect.width), std::max(rhs.rect.height, rhs.rect.width));

    cv::Vec2f diff = centroid - rhs.centroid;
    float dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
    if (dist > dist_thresh) 
        return false;

    float height_ratio = float(std::max(rhs.rect.height, rect.height)) / float(std::min(rhs.rect.height, rect.height));

    float top1 = rect.tl().y;
    float bottom1 = rect.br().y;
    float top2 = rhs.rect.tl().y;
    float bottom2 = rhs.rect.br().y;

    bool nonoverlap = top1 > bottom2 || top2 > bottom1;
    float ver_overlap = std::min(bottom1, bottom2) - std::max(top1, top2);
    float maxh = std::max(rect.height, rhs.rect.height);
    ver_overlap /= maxh;

    //std::cout << hor_overlap << " " << height_ratio << std::endl;
    //cv::Mat img1(rect.height+1, rect.width+1, CV_8UC1, cv::Scalar(0));
    //cv::Mat img2(rhs.rect.height+1, rhs.rect.width+1, CV_8UC1, cv::Scalar(0));
    //for (cv::Point pt : pixels) {
    //    img1.at<unsigned char>(pt.y - rect.y, pt.x - rect.x) = 255;
    //}
    //for (cv::Point pt : rhs.pixels) {
    //    img2.at<unsigned char>(pt.y - rhs.rect.y, pt.x - rhs.rect.x) = 255;
    //}
    //cv::imshow("GRP1", img1);
    //cv::imshow("GRP2", img2);
    //cv::waitKey(0);

    return ver_overlap > ConfigurationManager::instance()->get_minimum_vertical_overlap() && 
           height_ratio <= ConfigurationManager::instance()->get_maximum_height_ratio();
}

float CC::distance(const CC &rhs) const
{
    cv::Vec2f dist = centroid - rhs.centroid;
    float maxh = std::max(rect.height, rhs.rect.height);
    return sqrt(dist[0] * dist[0] + dist[1] * dist[1]) / maxh;
}

void CC::draw(cv::Mat img) const
{
    for (cv::Point p : pixels) {
        img.at<unsigned char>(p.y, p.x) = 255;
    }
}

float CCGroup::distance(const CCGroup &grp, const cv::Mat &distance_matrix) const
{
    float min = FLT_MAX;
    for (size_t i = 0; i < ccs.size(); i++) {
        for (size_t j = 0; j < grp.ccs.size(); j++) {
            if (ccs[i].can_link(grp.ccs[j])) {
                float d = distance_matrix.at<float>(ccs[i].id, grp.ccs[j].id);
                min = std::min(min, d);
            }
        }
    }
    return min;
}

bool CCGroup::is_overlapping_component_bbx(const CCGroup &grp, float thresh) const
{
    for (int i = 0; i < ccs.size(); i++) {
        for (int j = 0; j < grp.ccs.size(); j++) {
            if (ccs[i].id == grp.ccs[j].id) continue;
            cv::Rect intersect = ccs[i].rect & grp.ccs[j].rect;
            cv::Rect u = ccs[i].rect | grp.ccs[j].rect;
            float a1 = ccs[i].rect.width * ccs[i].rect.height;
            float a2 = grp.ccs[j].rect.width * grp.ccs[j].rect.height;
            float a_intersect = intersect.width * intersect.height;
            float a_union = u.width * u.height;
            if (a_intersect / a_union > thresh) {
                return true;
            }
        }
    }
    return false;
}

bool CCGroup::share_elements(const CCGroup &grp) const
{
    int nshared = 0;
    for (int i = 0; i < ccs.size(); i++) {
        for (int j = 0; j < grp.ccs.size(); j++) {
            if (ccs[i].id == grp.ccs[j].id) {
                nshared++;
            }
        }
    }
    return nshared == ccs.size() - 1;
}
bool CCGroup::eq(const CCGroup &grp) const 
{
    if (ccs.size() != grp.ccs.size()) return false;
    for (int i = 0; i < ccs.size(); i++) {
        bool found = false;
        for (int j = 0; j < grp.ccs.size(); j++) {
            if (ccs[i].id == grp.ccs[j].id) {
                found = true;
                break;
            }
        }
        if (found == false) return false;
    }
    return true;
}
bool CCGroup::can_link(const CCGroup &grp, const cv::Mat &distance_matrix) const
{
    if (ccs.size() != grp.ccs.size()) return false;
    if (!share_elements(grp)) return false;
    // check for overlap
    return !is_overlapping_component_bbx(grp, 0.8);
}

void CCGroup::link(const CCGroup &grp)
{
    //ccs.reserve(ccs.size() + grp.ccs.size());
    for (size_t i = 0; i < grp.ccs.size(); i++) {
        int id = grp.ccs[i].id;
        if (std::find_if(ccs.begin(), ccs.end(), [id] (const CC &cc) -> bool { return cc.id == id; }) == ccs.end())
            ccs.push_back(grp.ccs[i]);
    }
}

float CCGroup::get_bounding_box_area() const
{
    float result = 0.0f;
    for (CC c : ccs) {
        result += c.rect.width * c.rect.height;
    }
    return result;
}


float CCGroup::get_intersection_area(const CCGroup &grp) const 
{
    // intersect each own character w/ each character from grp
    float result = 0.0f;
    for (CC c_i : ccs) {
        for (CC c_j : grp.ccs) {
            cv::Rect intersection = c_i.rect & c_j.rect;
            result += intersection.width * intersection.height;
        }
    }
    return result;
}

cv::Rect CCGroup::get_rect() const
{
    cv::Rect result = ccs[0].rect;
    for (size_t i = 1; i < ccs.size(); i++) {
        result = result | ccs[i].rect;
    }
    return result;
}

bool CCGroup::is_significant_member(int idx) const
{
    cv::Rect my_rect = get_rect();
    cv::Rect other_rect = get_rect_without_element(idx);
    cv::Rect intersect = my_rect & other_rect;
    return intersect.width * intersect.height / (my_rect.width * my_rect.height) < 0.9;
}

cv::Rect CCGroup::get_rect_without_element(int leave_out) const
{
    if (ccs.size() == 1) {
        return cv::Rect();
    }

    int start = 0;
    if (leave_out == 0)
        start = 1;
    cv::Rect result = ccs[start].rect;
    for (size_t i = start+1; i < ccs.size(); i++) {
        if (leave_out == i) continue;
        result = result | ccs[i].rect;
    }
    return result;
}

std::vector<size_t> CCGroup::sorted_cc_indices() const
{
    std::vector<size_t> results(ccs.size());
    for (size_t i = 0; i < ccs.size(); i++) results[i] = i;

    std::sort(results.begin(), results.end(), [this] (size_t a, size_t b) -> bool {
        return ccs[a].rect.x < ccs[b].rect.x;
    });
    return results;
}

cv::Size CCGroup::group_size() const
{
    if (ccs.empty())
        return cv::Size(0,0);

    cv::Point pt(ccs[0].rect.br());
    for (size_t i = 1; i < ccs.size(); i++) {
        pt.x = std::max(ccs[i].rect.br().x, pt.x);
        pt.y = std::max(ccs[i].rect.br().y, pt.y);
    }
    return cv::Size(pt.x+1, pt.y+1);
}

cv::Mat CCGroup::get_image() const
{
    cv::Size size(group_size());
    cv::Mat img(size.height, size.width, CV_8UC3, cv::Scalar(255,255,255));
    for (CC c : ccs) {
        for (cv::Point p : c.pixels) {
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0,0,0);
        }
    }
    return img;
}

cv::Mat CCGroup::get_image(const cv::Size &size) const
{
    cv::Mat img(size.height, size.width, CV_8UC1, cv::Scalar(0));
    for (CC c : ccs) 
        c.draw(img); 
    return img;
}

float CCGroup::calculate_probability(const std::vector<double> &probs) const
{
    float result = 0.0f;
    for (CC cc : ccs) {
        result += probs[cc.feature_id];
    }
    return result / ccs.size();
}

bool CCGroup::is_overlapping(const CCGroup &other_group, float thresh) const
{
    cv::Rect my_rect = get_rect();
    cv::Rect other_rect = other_group.get_rect();

    cv::Rect intersect = my_rect & other_rect;
    cv::Rect union_rect = my_rect | other_rect;
    float my_area = my_rect.width * my_rect.height;
    float other_area = other_rect.width * other_rect.height;
    float intersect_area = intersect.width * intersect.height;
    float union_area = union_rect.width * union_rect.height;
    return intersect_area / union_area > thresh ||
           intersect_area / my_area > thresh ||
           intersect_area / other_area > thresh;
}

bool CCGroup::contains(const CCGroup &other_group) const
{
    for (size_t i = 0; i < other_group.ccs.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < ccs.size(); j++) {
            if (other_group.ccs[i].id == ccs[j].id) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

cv::Mat CCGroup::draw(const cv::Size &image_size, const std::vector<double> &probs, const cv::Mat &input, bool show) const
{
    cv::Mat result(input);
    if (result.empty()) {
        result = cv::Mat(image_size.height, image_size.width, CV_8UC3, cv::Scalar(255,255,255));
    } 
    for (CC c : ccs) {
        float prob = probs[c.feature_id];
        int greenish = 255 * prob;
        for (cv::Point px : c.pixels) {
            result.at<cv::Vec3b>(px.y, px.x) = cv::Vec3b(0, greenish, 255-greenish);
        }
    }
    cv::Rect r = get_rect();
    cv::rectangle(result, r.tl(), r.br(), cv::Scalar(255,0,0), 5);
    if (show) {
        cv::imshow("GROUP", result);
        cv::waitKey(0);
    }
    return result;
}

}
