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

#ifndef HIERARCHICALMSER_H

#define HIERARCHICALMSER_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

/*!
 Hierarchical Maximal Stable Extremal Regions class.

 The class implements Hierarchical MSER algorithm introduced by J. Matas.
 Unlike SIFT, SURF and many other detectors in OpenCV, this is salient region detector,
 not the salient point detector.

 It returns the regions, each of those is encoded as a contour.
*/
namespace cv {
class CV_EXPORTS_W HierarchicalMSER : public FeatureDetector
{
public:
    //! the full constructor
    // 1, 1, 14400, 0.5, 0.1
    // 1, 1, 14400000, 0.5, 0.01
    CV_WRAP explicit HierarchicalMSER( int _delta=1, int _min_area=1, int _max_area=14400000,
          double _max_variation=0.5, double _min_diversity=0.01, bool _stable_check=true );

    //! the operator that extracts the MSERs from the image or the specific part of it
    CV_WRAP_AS(detect) void operator()( const Mat& image, CV_OUT vector<vector<Point> >& msers,
                                        CV_OUT vector<double> &vars,
                                        CV_OUT vector<Vec4i> &hierarchie,
                                        const Mat& mask=Mat() ) const;
    AlgorithmInfo* info() const {return NULL;}

protected:
    void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    bool stable;
    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
};
}

#endif /* end of include guard: HIERARCHICALMSER_H */
