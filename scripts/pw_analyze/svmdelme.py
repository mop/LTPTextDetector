import numpy as np

from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

data = np.genfromtxt('dists_cleaned.csv', delimiter=',')

#samples = np.bitwise_or(np.bitwise_and(data[:,0] == -1, data[:,1] <= 2), data[:,0]==1)
#data = data[samples,:]
#data = data[data[:,7]>=5,:]
pos = data[data[:,0]==1,:]
neg = data[data[:,0]==-1,:]


a = 0.33974138  
b = 0.47850904 
c = -0.56307525


xrg = np.linspace(0,5,100)
yrg = np.linspace(0,5,100)

X,Y = np.meshgrid(xrg,yrg)
Z = a * X + b * Y + c
print Z

plt.scatter(neg[:,3], neg[:,4], color='r')
plt.scatter(pos[:,3], pos[:,4], color='b')
plt.contour(X,Y,Z)

plt.xlabel('medians')
plt.ylabel('heights')
plt.show()
#data[data[:,0]==1,0] = 2
#data[data[:,0]==-1,0] = 1
#data[data[:,0]==2,0] = -1

#print data.shape
#data = data[data[:,1]>0,:]
#data = data[data[:,2]>0,:]
#print data.shape
cv = ShuffleSplit(data.shape[0], n_iter=10, random_state=4)

min_dists = [1,2,3,4,5,100]
c1_grid = np.logspace(0,2,10)
c2_grid = np.logspace(0,4,15)
class_weights = {1:5, -1:1}
beta = 1.0


#best_params = {}
#best_fscore = 0
#for d in min_dists:
#    for C1 in c1_grid:
#        for C2 in c2_grid:
#            precisions = []
#            recalls = []
#            fscores = []
#            accs = []
#            for (train_idx, test_idx) in cv:
#                X = data[train_idx,3:5]
#                y = data[train_idx,0]
#
#                svm1 = LinearSVC(random_state=42, C=C1, class_weight=class_weights)
#                svm1.fit(X,y)
#
#                X_simple = data[train_idx, 4]
#                X_simple = X_simple.reshape((train_idx.shape[0], 1))
#                svm2 = LinearSVC(random_state=42, C=C2, class_weight=class_weights)
#                svm2.fit(X_simple, y)
#
#                #ys = svm2.predict(data[test_idx, 4].reshape((test_idx.shape[0], 1)))
#                #accs.append(svm2.score(data[test_idx, 4].reshape((test_idx.shape[0], 1)), data[test_idx,0]))
#
#                ys = np.zeros((test_idx.shape[0],))
#                for i,idx in enumerate(test_idx):
#                    if data[idx,-1] >= d:
#                        ys[i] = svm1.predict(data[idx, 3:5].reshape((1,2)))
#                    else:
#                        ys[i] = svm2.predict(data[idx, 4].reshape((1,1)))
#                
#                acc = np.sum(data[test_idx,0] == ys) / float(test_idx.shape[0])
#                accs.append(acc)
#                ps, rs, fs, ss = precision_recall_fscore_support(data[test_idx,0], ys, beta=beta)
#                precisions.append(ps[1])
#                recalls.append(rs[1])
#                fscores.append(fs[1])
#            print 'C1: %f, C2: %f, d: %f, prec: %f, recall: %f, f-score: %f, acc: %f' % (C1, C2, d, np.mean(precisions), np.mean(recalls), np.mean(fscores), np.mean(accs))
#            if np.mean(fscores) > best_fscore:
#                print '*'
#                best_fscore = np.mean(fscores)
#                best_params = {
#                    'C1': C1,
#                    'C2': C2,
#                    'd': d
#                }
#            

#best_params = {}
#best_fscore = 0
#print data.shape
#for C1 in c1_grid:
#    precisions = []
#    recalls = []
#    fscores = []
#    accs = []
#    for (train_idx, test_idx) in cv:
#        X = data[train_idx,3:5]
#        y = data[train_idx,0]
#
#        svm1 = LinearSVC(random_state=4,C=C1, class_weight=class_weights)
#        svm1.fit(X,y)
#
#        ys = np.zeros((test_idx.shape[0],))
#        for i,idx in enumerate(test_idx):
#            ys[i] = svm1.predict(data[idx, 3:5].reshape((1,2)))
#        
#        acc = np.sum(data[test_idx,0] == ys) / float(test_idx.shape[0])
#        accs.append(acc)
#        ps, rs, fs, ss = precision_recall_fscore_support(data[test_idx,0], ys, beta=beta)
#        precisions.append(ps[1])
#        recalls.append(rs[1])
#        fscores.append(fs[1])
#    svm = LinearSVC(random_state=4,C=C1, class_weight=class_weights)
#    svm.fit(data[:,3:5], data[:,0])
#    print 'C1: %f, prec: %f, recall: %f, f-score: %f, acc: %f' % (C1, np.mean(precisions), np.mean(recalls), np.mean(fscores), np.mean(accs))
#    print svm.coef_
#    print svm.intercept_
#    if np.mean(fscores) > best_fscore:
#        print '*'
#        best_fscore = np.mean(fscores)
#        best_params = {
#            'C1': C1
#        }
        


#print 'best:'
#print best_params

#svm = SVC(kernel='linear', C=best_params['C1'], class_weight=class_weights)
#svm.fit(data[:,1:3], data[:,0])
#print svm.coef_
#print svm.intercept_
#svm = SVC(kernel='linear', C=best_params['C1'], class_weight=class_weights)
#svm.fit(data[:,3:5], data[:,0])
#print svm.coef_
#print svm.intercept_

#svm = SVC()
#search = GridSearchCV(svm, {
#    'kernel': ('rbf',), 'C': [1,10,100,1000],
#    'gamma': [0.05, 0.1, 0.25, 0.128, 0.5, 1.0, 1.5],
#    'class_weight': ({1:1,-1:1},),
#    }, cv=cv)
svm = LinearSVC()
search = GridSearchCV(svm, {
    'C': np.logspace(0,4,15).tolist(),
    'class_weight': (
        {1:1,1:1},
        {1:1,-1:2},
        {1:1,-1:3},
        {1:1,-1:4},
        {-1:1,1:2},
        {-1:1,1:3},
        {-1:1,1:5},{1:4,-1:1},{1:5,-1:1})
    }, cv=cv, refit=False)

search.fit(data[:,3:5], data[:,0],cv=cv)

print search
print search.best_score_
print search.best_params_

svm = LinearSVC(random_state=42,**search.best_params_)
svm.fit(data[:,1:3],data[:,0])
print svm.score(data[:,1:3],data[:,0])
print svm.coef_
print svm.intercept_
