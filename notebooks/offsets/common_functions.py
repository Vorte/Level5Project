import numpy as np
import math
from sklearn import linear_model, cross_validation

def perform_CV(points, targets_x, targets_y, point_centers, n_folds=10):
    foldno = 1
    mse_x = []
    mse_y = []
    kf = cross_validation.KFold(len(points), n_folds=n_folds, shuffle=True)
    within_before = []
    within_after = []

    for train_index, test_index in kf:
        print
        print ("##### Fold %d #####" %foldno)
        print
        
        points_train, points_test = points[train_index], points[test_index]    
        t_x_train, t_x_test = targets_x[train_index], targets_x[test_index]
        t_y_train, t_y_test = targets_y[train_index], targets_y[test_index]
        
        regr_x = linear_model.LinearRegression()
        regr_y = linear_model.LinearRegression()    
        
        regr_x.fit(points_train, t_x_train)
        regr_y.fit(points_train, t_y_train)
        
        true, pred_x = t_x_test, regr_x.predict(points_test)
        
        mse = np.mean((pred_x - true) ** 2)
        print("MSE on x: %.2f" %mse)
        mse_x.append(mse)
        
        true, pred_y = t_y_test, regr_y.predict(points_test)
        
        mse = np.mean((pred_y - true) ** 2)
        print("MSE on y: %.2f" %mse)
        mse_y.append(mse)

        centers_test = point_centers[test_index]
        before = circle_button_error(points_test, centers_test)
        within_before.append(before)
        
        test_T = points_test.T
        x = np.array(test_T[0])+pred_x
        y = np.array(test_T[1])+pred_y

        corrected_points = np.array([x, y]).T    
        after = circle_button_error(corrected_points, centers_test)
        within_after.append(after)
            
        foldno += 1
    
    within_before = np.mean(np.array(within_before), 0)
    within_after = np.mean(np.array(within_after), 0)
       
    return np.array(mse_x), np.array(mse_y), within_before, within_after


def within_distance(point1, point2, distance):
    dot_pitch = 0.101195219 # 0.1011
    dist_px = int(distance/dot_pitch)
    
    dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    if dist <= dist_px:
        return True
    return False


def within_actual_button(point, center):
    width = 43
    height = 73
    if abs(point[0]-center[0])> height/2 or abs(point[1]-center[1])>width/2:
        return False
    return True
  
def circle_button_error(points, centers):
    no_points = []
    distances = range(1,11)

    for dist in distances:
        count = 0.0
        for i in range(len(points)):
            point = points[i]
            center = centers[i]
            if within_distance(point, center, dist):
                count += 1
#            if within_actual_button(point, center):
#                count += 1
                
        no_points.append(count/len(points))
        
    return no_points             
  
  
  
  
  
    
    

