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
#ifndef CACHEMANAGER_H

#define CACHEMANAGER_H

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace boost {
    namespace serialization {
        template<class Archive >
        inline void serialize(
            Archive &ar,
            cv::Point &p,
            const unsigned int file_version){
            ar & p.x;
            ar & p.y;
        }

        template<class Archive>
        inline void serialize(
            Archive & ar,
            cv::Vec4i &v,
            const unsigned int file_version) {
            ar & v[0];
            ar & v[1];
            ar & v[2];
            ar & v[3];
        }
    }
}


class CacheManager {
public:
    CacheManager(const std::string &dirname);
    ~CacheManager() {}
    //! Sets the global cache instance
    static void set_instance(std::shared_ptr<CacheManager> instance) { CacheManager::_instance = instance; }
    //! Retuns the global cache instance
    static std::shared_ptr<CacheManager> instance() { return CacheManager::_instance; }

    void load_image(const std::string &image_name);
    void save_image(const std::string &image_name);

    bool has_entry(int cc_id) const;
    bool has_feature_entry(int cc_id) const;
    std::vector<double> query(int cc_id) const;
    std::vector<float> query_feature(int cc_id) const;
    void set(int cc_id, const std::vector<double> &v);
    void set_feature(int cc_id, const std::vector<float> &f);
    void set_msers(const std::vector<std::vector<cv::Point> > &msers, const std::vector<cv::Vec4i> &hierarchy, int uid_offset) 
    {
        auto it = std::find(_uid_offsets.begin(), _uid_offsets.end(), uid_offset);
        if (it == _uid_offsets.end()) {
            _uid_offsets.push_back(uid_offset);
            _msers.push_back(msers);
            _hierarchy.push_back(hierarchy);
        } else {
            int idx = std::distance(_uid_offsets.begin(), it);
            _msers[idx] = msers;
            _hierarchy[idx] = hierarchy;
        }
    }
    bool get_msers(std::vector<std::vector<cv::Point> > &msers, std::vector<cv::Vec4i> &hierarchy, int uid_offset)
    {
        if (_msers.empty()) return false;
        auto it = std::find(_uid_offsets.begin(), _uid_offsets.end(), uid_offset);
        if (it == _uid_offsets.end())
            return false;
        int idx = std::distance(_uid_offsets.begin(), it);
        msers = _msers[idx];
        hierarchy = _hierarchy[idx];
        return true;
    }
private:
    static std::shared_ptr<CacheManager> _instance;
    std::string _dirname;
    //! Stroke-Width cache for unary features
    std::vector<std::vector<double> > _cache;
    //! Stroke-Width cache for unary features
    std::vector<std::vector<float> > _feature_cache;
    //! Msers
    std::vector<std::vector<std::vector<cv::Point> > > _msers;
    //! hierarchy
    std::vector<std::vector<cv::Vec4i> > _hierarchy;
    //! UID offsets for MSERS
    std::vector<int> _uid_offsets;
};

#endif /* end of include guard: CACHEMANAGER_H */
