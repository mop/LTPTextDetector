import numpy as np
from sklearn import svm
from sklearn import preprocessing, cross_validation
from sklearn.metrics import precision_recall_fscore_support

import os
import argparse

def do_train(rbf_svm, results, scores, scores_train, train_idx, test_idx, data, i):
    train = data[train_idx,:]
    test  = data[test_idx,:]

    train_data   = train[:,1:]
    train_labels = train[:,0]

    test_data   = test[:,1:]
    test_labels = test[:,0]

    rbf_svm.fit(train_data, train_labels)

    raw_dists = rbf_svm.decision_function(test_data)
    predictions = rbf_svm.predict(test_data)
    predictions_train = rbf_svm.predict(train_data)
    #predictions = np.array(pred_labels[:,1] >= pred_labels[:,0], dtype='float64')
    #predictions = predictions * 2 - 1.0
    #pred_labels = pred_labels[:,1]
    p,r,f,s = precision_recall_fscore_support(test_labels, predictions, beta=1.0)
    print p
    print r
    print f
    results[test_idx,0] = raw_dists[:,0]
    scores[i,0] = p[1]
    scores[i,1] = r[1]
    scores[i,2] = f[1]

    p,r,f,s = precision_recall_fscore_support(train_labels, predictions_train, beta=1.0)
    scores_train[i,0] = p[1]
    scores_train[i,1] = r[1]
    scores_train[i,2] = f[1]


def to_libsvm(data):
    result = []
    for i in xrange(data.shape[0]):
        line = ["%d:%f" % (idx+1,val) for (idx,val) in enumerate(data[i,1:])]
        line = ' '.join(line)
        line = '%d %s' % (data[i,0], line)
        result.append(line)
    return '\n'.join(result)

def main():
    parser = argparse.ArgumentParser(description='Process data with libsvm')
    parser.add_argument('--input', help='the input file')
    parser.add_argument('--output-norm', help='the output of the normalized file')
    parser.add_argument('--output-cv-results', help='the output of the normalized file')
    parser.add_argument('--output-libsvm', help='the output file for the data in libsvm format')
    parser.add_argument('--output-means', help='the output of the means')
    parser.add_argument('--output-stds', help='the output of the standard deviations')
    parser.add_argument('--gamma', help='gamma parameter of the rbf svm')
    parser.add_argument('--c', help='the c parameter of the rbf svm')
    parser.add_argument('--w1', help='the weight parameter')

    args = parser.parse_args()
    if None in [args.input, args.output_norm, args.output_libsvm, args.output_means, args.output_stds, args.output_cv_results]:
        parser.print_help()
        return
    print args.w1
    print args.gamma
    w1 = float(args.w1 or 4)
    gamma = float(args.gamma or 0.25)
    C = float(args.c or 32)

    print w1
    print gamma
    print C

    data = np.genfromtxt(args.input, delimiter=',')
    #data = np.delete(data, [1,7,8,9,10], axis=1)
    data[:,1] = np.log(data[:,1] + 1e-5)
    data[:,2] = np.log(data[:,2] + 1e-5)
    data[:,10] = np.log(data[:,10] + 1e-5)
    data[:,11] = np.log(data[:,11] + 1e-5)
    print np.any(np.isnan(data))
    print np.any(np.isinf(data))

    means = np.mean(data[:,1:], axis=0)
    data[:,1:] = data[:,1:] - means
    stds = np.std(data[:,1:], axis=0)
    data[:,1:] = data[:,1:] / stds

    rbf_svm = svm.SVC(kernel='rbf', gamma=gamma, C=C, probability=False, class_weight={-1:1,1:w1})

    nfolds = 10
    cv = cross_validation.ShuffleSplit(data.shape[0], n_iter=nfolds, random_state=42)

    results = np.zeros((data.shape[0],1), dtype='float64')
    scores = np.zeros((nfolds+1, 3), dtype='float64')
    scores_train = np.zeros((nfolds+1, 3), dtype='float64')

    i = 0
    for (train_idx, test_idx) in cv:
        print "Fold: %d" % i
        do_train(rbf_svm, results, scores, scores_train, train_idx, test_idx, data, i)
        i += 1

    # fixup the missing stuff
    test_idx  = np.nonzero(results[:,0] == 0.0)[0]
    train_idx = np.nonzero(results[:,0] != 0.0)[0]
    do_train(rbf_svm, results, scores, scores_train, train_idx, test_idx, data, i)

    print scores
    print 'precision: %f' % np.mean(scores[:,0])
    print 'recall: %f' % np.mean(scores[:,1])
    print 'fscore: %f' % np.mean(scores[:,2])

    print scores_train
    print 'train precision: %f' % np.mean(scores_train[:,0])
    print 'train recall: %f' % np.mean(scores_train[:,1])
    print 'train fscore: %f' % np.mean(scores_train[:,2])

    np.savetxt(args.output_cv_results, results, delimiter=',', fmt='%f')

    np.savetxt(args.output_norm, data, delimiter=',')
    np.savetxt(args.output_means, means.reshape((1,-1)), delimiter=',')
    np.savetxt(args.output_stds, stds.reshape((1,-1)), delimiter=',')

    with open(args.output_libsvm, 'w') as fp:
        fp.write(to_libsvm(data))

if __name__ == '__main__':
    main()
