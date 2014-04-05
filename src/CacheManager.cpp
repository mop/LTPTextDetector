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
#include <text_detector/CacheManager.h>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
namespace fs = boost::filesystem;


std::shared_ptr<CacheManager> CacheManager::_instance;
CacheManager::CacheManager(const std::string &dirname)
: _dirname(dirname)
{}

void CacheManager::load_image(const std::string &image_name)
{
    fs::path p(_dirname);
    p += "/";
    p += image_name;
    p += ".txt";

    _cache.clear();
    _feature_cache.clear();
    _msers.clear();
    _hierarchy.clear();
    _uid_offsets.clear();

    if (!fs::exists(p.generic_string())) return;

    std::ifstream ifs(p.generic_string());
    boost::archive::text_iarchive ia(ifs);
    ia & _cache;
    ia & _feature_cache;
    ia & _msers;
    ia & _hierarchy;
    ia & _uid_offsets;
}

void CacheManager::save_image(const std::string &image_name)
{
    fs::path p(_dirname);
    p += "/";
    p += image_name;
    p += ".txt";

    std::ofstream ofs(p.generic_string());
    boost::archive::text_oarchive oa(ofs);
    oa & _cache;
    oa & _feature_cache;
    oa & _msers;
    oa & _hierarchy;
    oa & _uid_offsets;

    _cache.clear();
    _feature_cache.clear();
    _msers.clear();
    _hierarchy.clear();
    _uid_offsets.clear();
}

bool CacheManager::has_entry(int i) const { 
    return _cache.size() > i && !_cache[i].empty(); 
}

std::vector<double> CacheManager::query(int cc_id) const
{
    return _cache.at(cc_id);
}

std::vector<float> CacheManager::query_feature(int cc_id) const
{
    return _feature_cache.at(cc_id);
}

void CacheManager::set(int cc_id, const std::vector<double> &v)
{
    if (_cache.size() <= cc_id) {
        _cache.resize((cc_id+1) * 2);
    }
    _cache[cc_id] = v;
}

void CacheManager::set_feature(int cc_id, const std::vector<float> &f)
{
    if (_feature_cache.size() <= cc_id) {
        _feature_cache.resize((cc_id+1)*2);
    }
    _feature_cache[cc_id] = f;
}

bool CacheManager::has_feature_entry(int cc_id) const
{
    return _feature_cache.size() > cc_id && !_feature_cache[cc_id].empty();
}


