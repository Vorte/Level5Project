import numpy as np
import math, ast, GPy
from sklearn import metrics, cross_validation


postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}


def contains_spikes(values):
    for value in values:
        if value>40000:
            return True
    
    return False

def grid_search(X, y, lengthscales):
    
    scores = np.zeros(len(lengthscales))
    for i in range(len(lengthscales)):
        lengthscale = lengthscales[i]
        kf = cross_validation.KFold(len(y), n_folds=3, shuffle=True)
        score = []
        for train_index, test_index in kf:

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            y_train = y_train.reshape(y_train.size, 1)

            kernel= GPy.kern.RBF(24, variance=1., lengthscale=lengthscale)  
            m = GPy.models.GPClassification(X = X_train, Y = y_train, kernel=kernel)
            prob = m.predict(X_test)
            pred = map(lambda x: 1 if x[0]>0.5 else 0, prob[0])
            score.append(metrics.accuracy_score(y_test, pred))
            
        scores[i] = np.mean(np.array(score))

    return lengthscales[np.argmax(scores)] 
            
def createlist(string):
    return map(float, string.replace('(', '').replace(')', '').split(','))

def within_distance(point1, point2, distance):
    dot_pitch = 0.101195219 # 0.1011
    dist_px = int(distance/dot_pitch)
    
    dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    if dist <= dist_px:
        return True
    return False


def iscorrect(a, b):
    key_width = 43    
    dist = math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )
    
    if dist > 2*key_width:
        return False
    return True


def circle_button_error(points, centers):
    no_points = []
    distances = range(1,8)

    for dist in distances:
        count = 0.0
        for i in range(len(points)):
            point = points[i]
            center = centers[i]
            if within_distance(point, center, dist):
                count += 1
                
        no_points.append(count/len(points))
        
    return no_points    


def get_key_centers():
    data = {}
    
    filename = "../../Loggingapp/keylocations.txt"
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = map(lambda x: x.split(' '), lines)
        for item in lines:
            data[item[0]] = (float(item[1]), float(item[2]))
            
    return data

def process_twohand(userid, posture = 0):
    locations = []
    bods = []
    targets_x = []
    targets_y = []
    y = []
    touch_centers = [] 
    
    centers = get_key_centers()
    filenos = ["2", "6", "9"]
    
    for fileno in filenos:
        filename = "../../data/"+str(userid)+"_"+fileno+"up.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            for line in lines:
                letter = line[1]
                location = list(ast.literal_eval(line[3]))
                center = centers[letter]
                bod = createlist(line[-1])
                
                if not iscorrect(location, center):
                    continue                    
                if contains_spikes(bod):
                    continue
                
                if line[0] == "left":
                    y.append(0+posture)
                else:
                    y.append(1+posture)
                
                touch_centers.append(center)
                targets_x.append(center[0]-location[0])
                targets_y.append(center[1]-location[1])            
                locations.append(location)
                bods.append(bod)
                
    return locations, bods, targets_x, targets_y, y, touch_centers


def process_posture(userid, filenos, posture): 
    
    locations = []
    bods = []
    targets_x = []
    targets_y = []
    y = []
    touch_centers = [] 
    centers = get_key_centers()
    
    for fileno in filenos:
        filename = "../../data/"+str(userid)+"_"+fileno+"up.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            for line in lines:
                letter = line[0]
                location = list(ast.literal_eval(line[2]))
                center = centers[letter]
                bod = createlist(line[-1])
                
                if not iscorrect(location, center):
                    continue                    
                if contains_spikes(bod):
                    continue
                
                targets_x.append(center[0]-location[0])
                targets_y.append(center[1]-location[1])                
                touch_centers.append(center)
                locations.append(location)
                bods.append(bod)
                y.append(posture)
                
    return locations, bods, targets_x, targets_y, y, touch_centers




    
    
    
