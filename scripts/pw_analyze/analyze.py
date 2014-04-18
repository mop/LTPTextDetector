import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
import cv2

FILTER_NEG = True

data = np.genfromtxt('dists.csv', delimiter=',')
dct = {}
soft_pp_dct = {}
height_dct = {}
centroid_dist_dct = {}
for x in data:
    if x[3] < 0:
        continue
    key = '%d-%d-%d' % (x[0], x[3], 1 if x[4] > 0 else 0)
    if FILTER_NEG and (x[5] <= 0 or x[6] <= 0 or x[8] <= 0):
        continue
    if not key in dct:
        dct[key] = []
    if not key in height_dct:
        height_dct[key] = []
    if not key in soft_pp_dct:
        soft_pp_dct[key] = []
    if not key in centroid_dist_dct:
        centroid_dist_dct[key] = []
    dct[key].append(x[5])
    soft_pp_dct[key].append(x[6])
    centroid_dist_dct[key].append(x[7])
    height_dct[key].append(x[8])

result = []
for (k,v) in dct.iteritems():
    img_id, txt_line, lbl = [int(x) for x in k.split('-')]
    soft_pp_val = soft_pp_dct[k]
    height_val = height_dct[k]
    centroid_val = centroid_dist_dct[k]

    if lbl <= 0: 
        continue
    neg_key = '%d-%d-%d' % (img_id, txt_line, (0 if lbl == 1 else 1))

    neg_vals = np.asarray([])
    neg_heights = np.asarray([])
    neg_soft_pp_vals = np.asarray([])
    neg_centroid_vals = np.asarray([])
    if neg_key in dct:
        neg_vals = np.asarray(dct[neg_key])
    if neg_key in height_dct:
        neg_heights = np.asarray(height_dct[neg_key])
    if neg_key in soft_pp_dct:
        neg_soft_pp_vals = np.asarray(soft_pp_dct[neg_key])
    if neg_key in centroid_dist_dct:
        neg_centroid_vals = np.asarray(centroid_dist_dct[neg_key])

    pos_vals = np.asarray(v)
    pos_soft_pp_vals = np.asarray(soft_pp_val)
    pos_heights = np.asarray(height_val)
    pos_centroid_dists = np.asarray(centroid_val)

    all_data = pos_vals
    all_soft_pp_vals = pos_soft_pp_vals
    all_heights = pos_heights
    all_centroid_vals = pos_centroid_dists

    if neg_vals.shape[0] > 0:
        all_data = np.r_[neg_vals, all_data]
    if neg_heights.shape[0] > 0:
        all_heights = np.r_[neg_heights, all_heights]
        #all_heights = np.r_[pos_heights, all_heights]
    if neg_soft_pp_vals.shape[0] > 0:
        all_soft_pp_vals = np.r_[neg_soft_pp_vals, all_soft_pp_vals]
    if neg_centroid_vals.shape[0] > 0:
        all_centroid_vals = np.r_[neg_centroid_vals, all_centroid_vals]

    #if neg_vals.shape[0] + pos_vals.shape[0] <= 1:
    #    continue

    mean_height = np.mean(all_heights)
    median_val = np.median(all_data)
    median_soft_pp = np.median(all_soft_pp_vals)
    median_centroid_val = np.median(all_centroid_vals)

    #if neg_vals.shape[0] <= 0:
    #    continue

    v = []
    for (pos, neg, norm, norm_pos) in [
             (pos_vals, neg_vals, np.median(all_data)+1e-10, np.median(all_data)+1e-10),
             (pos_vals, neg_vals, mean_height, mean_height), 
             (pos_soft_pp_vals, neg_soft_pp_vals, median_soft_pp + 1e-10, median_soft_pp + 1e-10),
             (pos_soft_pp_vals, neg_soft_pp_vals, mean_height, mean_height),
             (pos_centroid_dists, neg_centroid_vals, mean_height, mean_height),
             (pos_centroid_dists, neg_centroid_vals, median_centroid_val + 1e-10, median_centroid_val + 1e-10)]:

        dat = np.r_[pos, neg]


        if dat.shape[0] >= 2 and neg.shape[0] > 0:
            temp, classified_points, centroids = cv2.kmeans(data=np.asarray([dat]).transpose().astype(np.float32), K=2, bestLabels=None,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01), attempts=5, 
                flags=cv2.KMEANS_PP_CENTERS) 
            #centroids, labels = kmeans2(dat, 2)
            sums1 = abs(centroids[0] - centroids[1]) / (norm)
        else:
            sums1 = None

        if pos.shape[0] >= 2:
            temp, classified_points, centroids = cv2.kmeans(data=np.asarray([pos]).transpose().astype(np.float32), K=2, bestLabels=None,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01), attempts=5, 
                flags=cv2.KMEANS_PP_CENTERS) 

            #centroids, labels = kmeans2(pos, 2)
            sums2 = abs(centroids[0] - centroids[1]) / (norm_pos)
        else:
            sums2 = None

        v.append((sums1, sums2))

        if all(pos == pos_vals) and norm == median_val + 1e-10 and sums2 > 1.0:
            print 'oh noes, I\'m an outlier:'
            print img_id
            print txt_line

            print sums2
            print pos
            print neg
            print '---'

    if all([x is not None for (x,y) in v]):
        result.append([1] + [x for (x,y) in v] + [all_heights.shape[0]])
    if all([y is not None for (x,y) in v]):
        result.append([-1] +  [y for (x,y) in v] + [all_heights.shape[0]])
    
    continue

    plt.boxplot([pos, neg])
    plt.title('%d %d %d' % (img_id, txt_line, lbl))
    plt.show()
    for p in pos:
        result.append([int(lbl) * 2 - 1, p, neg_vals.shape[0] + pos_vals.shape[0]])
ary = np.asarray(result)
np.savetxt('dists_cleaned.csv', ary, delimiter=',', fmt='%f')
