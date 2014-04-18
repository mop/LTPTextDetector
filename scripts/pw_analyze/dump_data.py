import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.svm import LinearSVC

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

class Textline(object):
    """docstring for Textline"""
    def __init__(self, in_word_distances, between_word_distances, heights):
        self.in_word_distances = in_word_distances
        self.between_word_distances = between_word_distances
        self.heights = heights

    @property
    def n_words(self):
        return len(self.between_word_distances)+1

def estimate_svm(textlines):
    svc = LinearSVC(C=10, random_state=1, class_weight={1:0.35})
    
    data = []
    for line in textlines:

        dat = np.r_[line.in_word_distances, line.between_word_distances]
        if dat.shape[0] < 2:
            continue

        _, _, centroids = cv2.kmeans(data=np.asarray([dat]).transpose().astype(np.float32), K=2, bestLabels=None,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001), attempts=5, 
            flags=cv2.KMEANS_PP_CENTERS) 

        diff = abs(centroids[0] - centroids[1])

        if line.n_words == 1:
            # single word
            data.append([1] + [diff / np.mean(line.heights), diff / (np.median(dat) + 1e-10)])
            continue

        #multi word
        data.append([-1] + [diff / np.mean(line.heights), diff / (np.median(dat) + 1e-10)])

        if len(line.in_word_distances) < 2:
            continue
        # create an artificial single word
        _, _, centroids = cv2.kmeans(data=np.asarray([line.in_word_distances]).transpose().astype(np.float32), K=2, bestLabels=None,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001), attempts=5, 
            flags=cv2.KMEANS_PP_CENTERS) 
        diff = abs(centroids[0] - centroids[1])
        data.append([1] + [diff / np.mean(line.heights), diff / (np.median(line.in_word_distances) + 1e-10)])
    data = np.array(data)
    svc.fit(data[:,1:], data[:,0])
    return svc

def predict_textline(svc, textline):
    data = np.r_[textline.in_word_distances, textline.between_word_distances]

    if len(data) <= 1:
        return 1 # single word
    _, _, centroids = cv2.kmeans(data=np.asarray([data]).transpose().astype(np.float32), 
            K=2, bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01), 
        attempts=5, 
        flags=cv2.KMEANS_PP_CENTERS) 
    diff = abs(centroids[0] - centroids[1])
    v = np.r_[diff / np.mean(textline.heights), diff / np.median(np.r_[textline.in_word_distances,
        textline.between_word_distances])]
    label = svc.predict(v)
    if label == 1:
        return 1 # single word

    # multi-word
    max_centroid = max(centroids[0], centroids[1])
    min_centroid = min(centroids[0], centroids[1])

    results = []
    for x in data:
        if abs(max_centroid - x) < abs(min_centroid - x):
            results.append(True)
        else:
            results.append(False)
    return results

def test_textlines(svc, textlines):
    errs = 0
    n_gaps = 0
    for line in textlines:
        n_gaps += len(line.between_word_distances) + len(line.in_word_distances)
        result = predict_textline(svc, line)
        if isinstance(result, list):
            gt = ([False] * len(line.in_word_distances)) + ([True] * len(line.between_word_distances))
            errs_dists = [0 if a == b else 1 for a,b in zip(result, gt)]
            errs += np.sum(errs_dists)

        else:   # single word prediction
            if line.n_words > 1:
                print 'svm error'
                errs += len(line.between_word_distances)
    return errs / float(n_gaps)
        
        
def get_cv_split(data, fold, n_folds=3):
    batch_size = len(data) / n_folds
    batch_start = batch_size * fold
    batch_end = batch_size * fold + batch_size
    train = []
    test = []
    for i in xrange(len(data)):
        if i < batch_start or i > batch_end:
            train.append(data[i])
        else:
            test.append(data[i])
    return (train, test)

def cv_textlines(textlines):
    random.shuffle(textlines)
    n_folds = 10

    errors = []
    for i in xrange(n_folds):
        train, test = get_cv_split(textlines, i, n_folds)
        svc = estimate_svm(train)
        errors.append(test_textlines(svc, test))
    print 'mean error: ', np.mean(errors)


textlines = []
for (k,v) in dct.iteritems():
    img_id, txt_line, lbl = [int(x) for x in k.split('-')]
    soft_pp_val = soft_pp_dct[k]
    height_val = height_dct[k]

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

    pos_soft_pp_vals = np.asarray(soft_pp_val)
    pos_heights = np.asarray(height_val)

    all_soft_pp_vals = pos_soft_pp_vals
    all_heights = pos_heights


    textlines.append(Textline(pos_soft_pp_vals, neg_soft_pp_vals, all_heights))
    continue

    if neg_heights.shape[0] > 0:
        all_heights = np.r_[neg_heights, all_heights]
    if neg_soft_pp_vals.shape[0] > 0:
        all_soft_pp_vals = np.r_[neg_soft_pp_vals, all_soft_pp_vals]

    #if neg_vals.shape[0] + pos_vals.shape[0] <= 1:
    #    continue

    mean_height = np.mean(all_heights)
    median_soft_pp = np.median(all_soft_pp_vals)

    #if neg_vals.shape[0] <= 0:
    #    continue

    v = []
    for (pos, neg, norm, norm_pos) in [
             (pos_soft_pp_vals, neg_soft_pp_vals, median_soft_pp + 1e-10, median_soft_pp + 1e-10),
             (pos_soft_pp_vals, neg_soft_pp_vals, mean_height, mean_height)]:

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

cv_textlines(textlines)
