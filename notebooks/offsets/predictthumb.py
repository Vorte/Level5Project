import numpy as np
import dataIO
from sklearn import cross_validation, linear_model, preprocessing, svm
from sklearn.grid_search import GridSearchCV

def learn_offset(points, targets):
    regr = linear_model.LinearRegression()
    regr.fit(points, targets)
    
    return regr


def run(userId):
	locations, bod, targets_x, targets_y, y, touch_centers = dataIO.process_twohand(userId)

	locations = np.array(locations)
	bod = np.array(bod)
	targets_x = np.array(targets_x)
	targets_y = np.array(targets_y)
	y = np.array(y)
	touch_centers = np.array(touch_centers)

	within_before = dataIO.circle_button_error(locations, touch_centers)
	locations = np.concatenate((locations, locations**2),1)

	within_after = []
	kf = cross_validation.KFold(len(y), n_folds=10, shuffle=True)

	for train_index, test_index in kf:
		
		points_train, points_test = locations[train_index], locations[test_index]
		bod_train, bod_test = bod[train_index], bod[test_index]
		t_x_train, t_x_test = targets_x[train_index], targets_x[test_index]
		t_y_train, t_y_test = targets_y[train_index], targets_y[test_index]
		y_train, y_test = y[train_index], y[test_index]
		centers_train, centers_test = touch_centers[train_index], touch_centers[test_index]
		
		scaler = preprocessing.StandardScaler().fit(bod_train)  
		bod_scaled = scaler.transform(bod_train)
		
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1 ,1e-2, 1e-3],
		                     'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]},
		                    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]}]
		
		#clf = GridSearchCV(svm.SVC(C=1, cache_size=500), tuned_parameters)
		clf = svm.SVC(C=1, kernel='rbf', cache_size=500, gamma=0.1)
		clf.fit(bod_scaled, y_train)
		
		regr_x = []
		regr_y = []
		
		for i in range(0,2):
		    index = np.where(y_train==i)[0]
		    regr_x.append(learn_offset(points_train[index], t_x_train[index]))
		    regr_y.append(learn_offset(points_train[index], t_y_train[index]))
		    
		new_points = np.zeros((len(points_test), 2))
		for i in range(len(points_test)):
		    
		    point = points_test[i]
		    bod_data = scaler.transform(bod_test[i])
		    pred = clf.predict(bod_data)
		    
		    pred_x = regr_x[pred].predict(point)
		    pred_y = regr_y[pred].predict(point)
		    
		    new_points[i][0] = point[0]+pred_x
		    new_points[i][1] = point[1]+pred_y
		 
		within_after.append(dataIO.circle_button_error(new_points, centers_test))
		new_points = np.array(new_points).T
		centers_test = centers_test.T
   
	within_after = np.mean(np.array(within_after), 0)
		
	return np.array(within_before), np.array(within_after)








