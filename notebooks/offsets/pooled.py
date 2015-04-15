import numpy as np
import dataIO
from sklearn import cross_validation, linear_model

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

    mse_x_b = np.mean(targets_x ** 2)
    mse_y_b = np.mean(targets_y ** 2)
    within_before = np.array(dataIO.circle_button_error(locations, touch_centers))
    locations = np.concatenate((locations, locations**2),1)

    se_x = []
    se_y = []
    within_after = []
    kf = cross_validation.KFold(len(y), n_folds=10, shuffle=True)

    for train_index, test_index in kf:

        points_train, points_test = locations[train_index], locations[test_index]
        t_x_train, t_x_test = targets_x[train_index], targets_x[test_index]
        t_y_train, t_y_test = targets_y[train_index], targets_y[test_index]
        centers_train, centers_test = touch_centers[train_index], touch_centers[test_index]

        regr_x = learn_offset(points_train, t_x_train)
        regr_y = learn_offset(points_train, t_y_train)

        new_points = np.zeros((len(points_test),2))
        for i in range(len(points_test)):
            point = points_test[i]

            pred_x = regr_x.predict(point)
            pred_y = regr_y.predict(point)

            new_points[i][0] = point[0]+pred_x
            new_points[i][1] = point[1]+pred_y

        within_after.append(dataIO.circle_button_error(new_points, centers_test))
        new_points = new_points.T
        centers_test = centers_test.T

        se_x.append((new_points[0]-centers_test[0])**2)
        se_y.append((new_points[1]-centers_test[1])**2)

    se_x = np.array([item for sublist in se_x for item in sublist])
    se_y = np.array([item for sublist in se_y for item in sublist])

    mse_x_a = np.mean(se_x)
    mse_y_a = np.mean(se_y)      
    within_after = np.mean(np.array(within_after), 0)
    
    return within_before, within_after, mse_x_b, mse_y_b, mse_x_a, mse_y_a



    
    