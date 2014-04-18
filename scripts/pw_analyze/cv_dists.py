import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt

def main():
    data = np.loadtxt('dists_cleaned.csv', delimiter=',')
    plt.boxplot(data[data[:,0]==1,3])
    plt.xlabel('height')

    svm = LinearSVC(random_state=1,penalty='l1', dual=False, C=1, class_weight='auto')
    cv = ShuffleSplit(data.shape[0], n_iter=data.shape[0], random_state=2)
    print cross_val_score(svm, data[:,1:2], data[:,0], cv=cv).mean()
    print data[:,2:3].shape
    svm.fit(data[:,2:3], data[:,0])
    print 'P', float(np.logical_and(svm.predict(data[:,2:3])  == -1, data[:,0] == -1).sum()) / ((svm.predict(data[:,2:3]) == -1).sum())
    print 'R', float(np.logical_and(svm.predict(data[:,2:3])  == -1, data[:,0] == -1).sum()) / (data[:,0] == -1).sum()
    print svm.coef_
    print svm.intercept_

if __name__ == '__main__':
    main()
