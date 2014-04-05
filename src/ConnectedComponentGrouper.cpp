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
#include <text_detector/ConnectedComponentGrouper.h>

#include <boost/timer/timer.hpp>

#include <text_detector/CCUtils.h>
#include <text_detector/UnionFind.h>

namespace TextDetector {

void ConnectedComponentGrouper::create_initial_groups(
		const std::vector<std::pair<int, std::vector<cv::Point> > >& comps,
		const std::vector<MserElement>& all_elements, std::vector<CC>& ccs,
		std::vector<CCGroup>& groups,
		std::vector<MserElement>& elements) const {
	for (int i = 0; i < comps.size(); i++) {
		CC cc(i, comps[i].first, comps[i].second);
		ccs.push_back(cc);
		CCGroup ccg;
		ccg.ccs.push_back(cc);
		groups.push_back(ccg);
		elements.push_back(all_elements[comps[i].first]);
	}
}

void ConnectedComponentGrouper::fill_distance_matrix(const std::vector<CC>& ccs,
		const std::vector<MserElement>& elements, const cv::Mat& train_image,
		const cv::Mat& gradient_image,
		cv::Mat& distance_matrix) const {

    // from liblinear: /train -w1 100 -c 1
    cv::Mat w = (cv::Mat_<float>(1,9) <<
        -0.1337192487596409,
        -0.5229476757516559,
        0.1838924512606428,
        -0.8868503369712512,
        -3.054642747162522,
        1.051516533973762,
        0.7878517158725902,
        0.4666205783808373,
        0.5530583456726066
    );
    const float bias = -0.6339231904073611;


    #pragma omp parallel for
	for (int i = 0; i < ccs.size(); i++) {
		const MserElement &el1 = elements[i];
		for (int j = i + 1; j < ccs.size(); j++) {
			if (i == j)
				continue;
			const MserElement &el2 = elements[j];
			cv::Vec2f diff = el1.get_centroid() - el2.get_centroid();
			double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1]);

			double thresh = 2
					* std::min(
							std::max(el1.get_bounding_rect().width,
									el1.get_bounding_rect().height),
							std::max(el2.get_bounding_rect().width,
									el2.get_bounding_rect().height));
			if (dist < thresh) {
				cv::Mat f = el1.compute_pairwise_features(train_image,
						gradient_image, el2);
				cv::Mat f2 = f.colRange(0, 9);
				assert(f2.cols == 9);
				float val = -(f2.dot(w) + bias);
				distance_matrix.at<float>(i, j) = val;
				distance_matrix.at<float>(j, i) = val;
				if (ConfigurationManager::instance()->ignore_grouping_svm()) {
					distance_matrix.at<float>(i, j) = -100;
					distance_matrix.at<float>(j, i) = -100;
				}
			} else {
				distance_matrix.at<float>(i, j) = FLT_MAX;
				distance_matrix.at<float>(j, i) = FLT_MAX;
			}
		}
	}
}

void ConnectedComponentGrouper::merge_components(
	const cv::Mat& distance_matrix,
    std::vector<CCGroup>& groups) const {

	while (true) {
		UnionFind groupings(groups.size());
		int any_found = 0;
		for (size_t i = 0; i < groups.size(); i++) {
			for (size_t j = i + 1; j < groups.size(); j++) {
				float dist = groups[i].distance(groups[j], distance_matrix);
				if (dist < _distance_threshold) {
					{
						groupings.union_set(i, j);
						any_found += 1;
						//CCGroup grp;
						//grp = groups[i];
						//grp.link(groups[j]);
						//std::vector<CCGroup> grps; grps.push_back(grp);
						//show_groups_color(grps, cv::Size(train_image.cols, train_image.rows), probs, true);
					}
				}
			}
		}
		if (!any_found)
			break;

		// merge groups (in reverse order!)
		std::vector<bool> root_groups(groups.size(), 0);
		int n_roots = 0;
		for (size_t i = 0; i < groups.size(); i++) {
			int set_id = groupings.find_set(i);
			if (set_id == i) {
				root_groups[i] = true;
				n_roots++;
				continue;
			}
			groups[set_id].link(groups[i]);
			std::vector<CCGroup> grp;
			grp.push_back(groups[set_id]);
		}
		std::vector<CCGroup> new_groups;
		new_groups.reserve(n_roots);
		for (size_t i = 0; i < groups.size(); i++) {
			if (root_groups[i])
				new_groups.push_back(groups[i]);
		}
		groups = new_groups;
	}
}

void ConnectedComponentGrouper::prune_low_probability_groups(
    const std::vector<double> &probs, std::vector<CCGroup> &groups) const
{
    float group_threshold = ConfigurationManager::instance()->get_word_group_threshold();
    int min_group_size = ConfigurationManager::instance()->get_min_group_size() - 1;
    groups.erase(std::remove_if(groups.begin(), groups.end(), [&probs, group_threshold, min_group_size] (const CCGroup &g) -> bool {
        float proba = 0.0f;
        // groups of size 2 are erased
        if (g.ccs.size() <= min_group_size) return true;

        cv::Rect rect = g.get_rect();
        // num ccs are the number of connected components which are 'important'.
        // A connected is important if its probability is over the group-threshold
        // or without the component the enclosing bounding box will get smaller
        int num_ccs = 0;
        for (size_t i = 0; i < g.ccs.size(); i++) {
            if (probs[g.ccs[i].feature_id] < group_threshold) {
                // check if it is fully contained in the bounding rect of the group
                // and it is not touching any edges
                if (g.is_significant_member(i)) {
                    num_ccs++;
                    proba += probs[g.ccs[i].feature_id];
                }
            } else {
                num_ccs++;
                proba += probs[g.ccs[i].feature_id];
            }
        }

        // sometimes, all ccs in groups are overlapping with each other -> fallback to
        // other heuristic
        if (num_ccs == 0)
            num_ccs = g.ccs.size();
        assert(num_ccs > 0);
        return (proba / num_ccs) < group_threshold;
    }), groups.end());

}

void ConnectedComponentGrouper::prune_overlapping_groups(
	const std::vector<double> &probs,
    std::vector<CCGroup> &groups) const
{
    std::vector<int> to_remove;
    for (int i = 0; i < groups.size(); i++) {
        cv::Rect r_i = groups[i].get_rect();
        float a_i = groups[i].get_bounding_box_area();
        float prob_i = groups[i].calculate_probability(probs);

        for (int j = i+1; j < groups.size(); j++) {
            if (i == j) continue;
            cv::Rect r_j = groups[j].get_rect();
            float a_j = groups[j].get_bounding_box_area();

            cv::Rect intersect = r_i & r_j;
            float area_intersect = intersect.width * intersect.height;
            if (area_intersect > 0) {
                float prob_j = groups[j].calculate_probability(probs);
                float a_both = groups[i].get_intersection_area(groups[j]);
                if (prob_i < prob_j) {
                    // it is intersecting more w/ group i
                    if (a_both / a_i > _overlap_threshold) {
                        to_remove.push_back(i);
                        break;
                    }
                } else {
                    // it is intersecting more w/ group j
                    if (a_both / a_j > _overlap_threshold) {
                        to_remove.push_back(j);
                        continue;
                    }
                }
            }
        }
    }

    if (to_remove.size() > 0) {
        if (ConfigurationManager::instance()->verbose())
            std::cout << "Removing: " << to_remove.size() << " Textlines" << std::endl;
    }

    std::vector<CCGroup> new_groups;
    new_groups.reserve(groups.size());
    for (int i = 0; i < groups.size(); i++) {
        if (std::find(to_remove.begin(), to_remove.end(), i) == to_remove.end()) {
            new_groups.push_back(groups[i]);
        }
    }
    groups = new_groups;
}

void ConnectedComponentGrouper::operator()(
		const cv::Mat &train_image,
        const cv::Mat &gradient_image,
        const std::vector<double> &probs,
        const std::vector<MserElement> &all_elements,
        const std::vector<std::pair<int, std::vector<cv::Point> > > &comps,
        std::vector<CCGroup> &groups) const
{

    std::vector<MserElement> elements;
    elements.reserve(comps.size());
    std::vector<CC> ccs;
	create_initial_groups(comps, all_elements, ccs, groups, elements);

    cv::Mat distance_matrix(
        std::max(1, static_cast<int>(ccs.size())),
        std::max(1, static_cast<int>(ccs.size())), CV_32FC1);

    if (ConfigurationManager::instance()->verbose()) {
        std::cout << "Computing distances for: " << ccs.size()
        		  << " connected compontents!" << std::endl;
    }
    boost::timer::cpu_timer t;
    t.start();
	fill_distance_matrix(ccs, elements, train_image,
		gradient_image, distance_matrix);

    if (ConfigurationManager::instance()->verbose()) {
        std::cout << "Computed distance-matrix in " <<
            boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    }

    t.start();
	merge_components(distance_matrix, groups);

    if (ConfigurationManager::instance()->verbose()) {
        std::cout << "Grouped components in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    }

    prune_low_probability_groups(probs, groups);
    prune_overlapping_groups(probs, groups);

    //show_groups_color(groups, cv::Size(train_image.cols, train_image.rows), probs, true);
    //show_groups(groups, cv::Size(train_image.cols, train_image.rows), true);
}

} /* namespace TextDetector */
