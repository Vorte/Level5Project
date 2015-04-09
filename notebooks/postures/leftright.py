import numpy as np
import dataIO
from sklearn import svm, preprocessing, cross_validation
from sklearn.grid_search import GridSearchCV

def run(userId):
	left, right = dataIO.read_twothumb(userId) # USERID

	X = np.array(left+right)
	y = np.array([0 for x in range(len(left))]+ [1 for x in range(len(right))])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

	scaler = preprocessing.StandardScaler().fit(X_train)  
	X_scaled = scaler.transform(X_train)
	test_scaled = scaler.transform(X_test)

	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1, 1e-2, 1e-3],
		               'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]},
		              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]}]


	clf = GridSearchCV(svm.SVC(C=1, cache_size=500), tuned_parameters) 
	clf.fit(X_scaled, y_train)

	y_true, y_pred = y_test, clf.predict(test_scaled)
	diff = y_true-y_pred
	return 1.0*np.nonzero(diff)[0].shape[0]/y_true.shape[0]


miscl = np.zeros(17)
for i in range(1,18):
	miscl[i-1] = run(i)

print ("Average misclassification rate %.1f %%" %(100*np.mean(miscl)))




