import numpy as np
import scipy.misc as misc

#p = '../Pictures/testImgs/'
#X = []
#X.append((misc.imread(p+'download.jpg').flatten()))
#X.append((misc.imread(p+'images.jpg').flatten()))
#X = np.asarray(X)
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1,2,2,1])

from sklearn.svm import SVC
import sklearn

clf = SVC()
clf.fit(X,y)

x_t = np.array([[2,2], [4,2]])

print(clf.decision_function(x_t))
print(np.argmin(clf.decision_function(x_t)))

