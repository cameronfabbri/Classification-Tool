import numpy as np
import scipy.misc as misc

#p = '../Pictures/testImgs/'
#X = []
#X.append((misc.imread(p+'download.jpg').flatten()))
#X.append((misc.imread(p+'images.jpg').flatten()))
#X = np.asarray(X)
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 2,2,1])

from sklearn.svm import SVC
import sklearn

#print X.shape
#print y.shape
#exit()
clf = SVC()
clf.fit(X,y)

print(clf.decision_function(X))
print(np.argmin(clf.decision_function(X)))

