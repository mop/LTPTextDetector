import numpy as np
from sklearn import svm
from sklearn import preprocessing, cross_validation
from scipy.optimize import fmin
import argparse

def calc_err(X,l,w):
    resp = X * w[0] + w[1]
    results = 1/(1+np.exp(-resp))
    return np.sum((results - l) ** 2)

def do_fit(train_idx, test_idx, result, data, resp, labels):
    X = resp[train_idx,:]
    if test_idx.shape[0] > 0:
        X_test = resp[test_idx,:]
    else:
        X_test = np.array([])
    l = labels[train_idx]

    pos = X[l == 1]
    neg = X[l == -1]
    #neg = neg[:pos.shape[0]]

    dat  = np.r_[pos,neg]
    ldat = np.r_[
        np.ones((pos.shape[0],1)) * (float(pos.shape[0]) + 1.0) / (float(pos.shape[0]) + 2.0), -np.ones((neg.shape[0],1)) * 1.0 / (float(neg.shape[0]) + 2.0)]

    min_res = fmin(lambda a: calc_err(dat,ldat,a), np.array([0.1, 0.0]))
    print min_res
    a = min_res
    if test_idx.shape[0] > 0:
        result[test_idx] = 1/(1+np.exp(-(a[0]*X_test + a[1])))
    return a
    
def main():
    parser = argparse.ArgumentParser(description='Fit probabilities to svm outputs')
    parser.add_argument('--input', help='the original input file')
    parser.add_argument('--input-svm-cv-results', help='the input of the cross validated svm results')
    parser.add_argument('--output-cv-results', help='the output of the cross-validated normalized file')
    parser.add_argument('--output-params', help='the output of the final parameter')

    args = parser.parse_args()
    if None in [args.input, args.input_svm_cv_results, args.output_cv_results, args.output_params]:
        parser.print_help()
        return

    data = np.genfromtxt(args.input, delimiter=',')
    resp = np.genfromtxt(args.input_svm_cv_results, delimiter=',')
    resp = resp.reshape((resp.shape[0], 1))

    labels = data[:,0]

    nfolds = 10
    result = np.zeros((data.shape[0], 1), dtype='float64')

    cv = cross_validation.ShuffleSplit(data.shape[0], n_iter=nfolds, random_state=42)
    for (train_idx, test_idx) in cv:
        do_fit(train_idx, test_idx, result, data, resp, labels)

    test_idx = np.nonzero(result == 0.0)[0]
    train_idx = np.nonzero(result != 0.0)[0]

    do_fit(train_idx, test_idx, result, data, resp, labels)

    np.savetxt(args.output_cv_results, result, fmt='%f', delimiter=',')

    a = do_fit(np.arange(result.shape[0]), np.array([]), np.zeros((result.shape[0], 1)), data, resp, labels)
    np.savetxt(args.output_params, a, fmt='%f', delimiter=',')

if __name__ == '__main__':
    main()
