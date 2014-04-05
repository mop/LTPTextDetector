import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='Cross validation for textline svm classify')
    parser.add_argument('--input', help='the original input file')

    args = parser.parse_args()
    if args.input is None:
        parser.print_help()
        return

    data = np.loadtxt(args.input, delimiter=',')
    X = data[:,6:]
    y = data[:,5]
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, random_state=42)
    c_grid = np.logspace(-1,2,4)
    w_grid = np.logspace(0,1,4)
    params = [(c,w) for c in c_grid for w in w_grid]
    random.shuffle(params)
    
    #for (c,w) in params:
    #    precisions = []
    #    recalls = []
    #    fscores = []
    #    for (train_idx, test_idx) in cv:
    #        s = svm.LinearSVC(C=c, class_weight={1: w, -1: 1})
    #        s.fit(X[train_idx,:], y[train_idx,:])
    #        results = s.predict(X[test_idx])
    #        (precision, recall, f1_score, support) = precision_recall_fscore_support(y[test_idx,:], results)
    #        precisions.append(precision[1])
    #        recalls.append(recall[1])
    #        fscores.append(f1_score[1])
    #    print 'C: %f, W: %f' % (c,w)
    #    print 'precision: ', np.mean(np.asarray(precisions))
    #    print 'recall: ', np.mean(np.asarray(recalls))
    #    print 'fscore: ', np.mean(np.asarray(fscores))

    s = svm.LinearSVC(C=10, class_weight={1: 10, -1: 1})
    s.fit(X, y)
    print s.coef_.T
    print s.intercept_


if __name__ == '__main__':
    main()
