import numpy as np
from sklearn import linear_model, cross_validation

def perform_CV(points, targets_x, targets_y, n_folds=10):
    foldno = 1
    mse_x = []
    mse_y = []
    kf = cross_validation.KFold(len(points), n_folds=n_folds, shuffle=True)
    
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
        
        true, pred = t_x_test, regr_x.predict(points_test)
        
        mse = np.mean((pred - true) ** 2)
        print("MSE on x: %.2f" %mse)
        mse_x.append(mse)
        
        true, pred = t_y_test, regr_y.predict(points_test)
        
        mse = np.mean((pred - true) ** 2)
        print("MSE on y: %.2f" %mse)
        mse_y.append(mse)
            
        foldno +=1
        
    return np.array(mse_x), np.array(mse_y)
    
    
    

