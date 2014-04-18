import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit

def main():
    data = np.loadtxt('dists.csv', delimiter=',')

    X = np.c_[data[:,4], data[:,6] / data[:,8]]

    svm = LinearSVC(random_state=1,penalty='l1', dual=False, C=100, class_weight='auto')
    cv = ShuffleSplit(data.shape[0], n_iter=data.shape[0], random_state=2)
    print cross_val_score(svm, X[:,1:], X[:,0], cv=cv).mean()

    pos = data[data[:,4]==1,:]
    neg = data[data[:,4]==-1,:]

    deltas = pos[:,6]
    heights = pos[:,8]
    f = deltas / heights

    f2 = neg[:,6] / neg[:,8]

    plt.boxplot([f, f2])
    plt.show()



if __name__ == '__main__':
    main()

