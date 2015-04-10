import numpy as np
import dataIO, GPy
from sklearn import cross_validation, linear_model, preprocessing

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
		y_train = y_train.reshape(y_train.size, 1)

		l = np.sqrt(1/(2*0.1))
		kernel= GPy.kern.RBF(24, variance=1., lengthscale=l)  
		m = GPy.models.GPClassification(X = bod_scaled, Y = y_train, kernel=kernel)
		
	#     m = GPy.models.GPClassification(X = bod_scaled, Y = y_train)
		
	#     l1 = m.rbf.lengthscale.values[0]
	#     v1 = m.rbf.variance.values[0]
	#     l2 = float("inf")
	#     v2 = float("inf")
	#     while abs(l1-l2)>0.01 or abs(v1-v2)>0.01:
	#         m.optimize('bfgs', max_iters=100)
	#         l1 = l2
	#         v1 = v2
	#         l2 = m.rbf.lengthscale.values[0]
	#         v2 = m.rbf.variance.values[0]
		
	#     print m
		
		regr_x = []
		regr_y = []
		
		for i in range(0,2):
		    index = np.where(y_train==i)[0]
		    regr_x.append(learn_offset(points_train[index], t_x_train[index]))
		    regr_y.append(learn_offset(points_train[index], t_y_train[index]))
		    
		new_points = np.zeros((points_test.shape[0],2))
		for i in range(len(points_test)):
		    
		    point = points_test[i]
		    bod_data = scaler.transform(bod_test[i])
		    bod_data = bod_data.reshape(1, 24)

		    if m.predict(bod_data)[0][0][0]>0.5:
		        pred = 1
		    else:
		        pred = 0
		        
		    pred_x = regr_x[pred].predict(point)
		    pred_y = regr_y[pred].predict(point)
		    
		    new_points[i][0] = point[0]+pred_x
		    new_points[i][1] = point[1]+pred_y
		 
		within_after.append(dataIO.circle_button_error(new_points, centers_test))
		new_points = np.array(new_points).T
		centers_test = centers_test.T
		
	within_after = np.mean(np.array(within_after), 0)

	return np.array(within_before), np.array(within_after)





    
