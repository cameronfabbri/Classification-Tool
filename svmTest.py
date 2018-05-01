import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

from sklearn.svm import SVC
import sklearn

clf = SVC()
clf.fit(X,y)

print(clf.decision_function(X))

