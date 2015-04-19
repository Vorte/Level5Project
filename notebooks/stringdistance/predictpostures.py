import numpy as np
import Levenshtein, ast, dataIO, VBGP
from sklearn import cross_validation, linear_model, preprocessing, svm, utils
from sklearn.grid_search import GridSearchCV

def error_rate(a, b):
    msd = Levenshtein.distance(a,b)
    
    return 100.0*msd/max(len(a), len(b))

def learn_offset(points, targets):
    regr = linear_model.LinearRegression()
    regr.fit(points, targets)
    
    return regr


postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}
                
def run(userId):
    keys = postures.keys()
    locations = []
    bod = []
    targets_x = []
    targets_y = []
    y = []

    posture = 0
    for key in keys:
        filenos = postures[key]
        if key == "two_hand":
            a, b, c, d, e, f = dataIO.process_twohand(userId, posture)
            posture += 2
        else:
            a, b, c, d, e, f = dataIO.process_posture(userId, filenos, posture)
            posture += 1

        locations += a
        bod += b
        targets_x += c 
        targets_y += d 
        y += e

    locations = np.array(locations)
    bod = np.array(bod)
    targets_x = np.array(targets_x)
    targets_y = np.array(targets_y)
    y = np.array(y)

    locations, bod, targets_x, targets_y, y = utils.shuffle(locations, bod, targets_x, targets_y, y)
    locations = np.concatenate((locations, locations**2),1)

    scaler = preprocessing.StandardScaler().fit(bod)  
    bod_scaled = scaler.transform(bod)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1 ,1e-2, 1e-3],
                         'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]}]

    #clf = GridSearchCV(svm.SVC(C=1, cache_size=500), tuned_parameters, n_jobs=-1)
    #clf = svm.SVC(C=100, kernel='rbf', gamma=0.01, cache_size=500)
    #clf.fit(bod_scaled, y)
    thetas = np.array([[1 for x in range(24)], [0.1 for x in range(24)], [0.01 for x in range(24)],
         [0.001 for x in range(24)],[10 for x in range(24)]])
    
    gp = VBGP.VBGP()
    gp.fit(bod_scaled, y, thetas[0], nos_its=3, thresh=0.1)
    gp.optimize(thetas, nos_its=3, thresh=0.1)

    regr_x = [] 
    regr_y = []

    
    for i in range(5):
        index = np.where(y==i)[0]
        regr_x.append(learn_offset(locations[index], targets_x[index]))
        regr_y.append(learn_offset(locations[index], targets_y[index])) 
        
    with open("../../Loggingapp/dataset.txt") as f:
        pool = f.read().splitlines()

    pool = sorted(pool, key=len, reverse=True)

    req_sentences = []
    typed_phys = []
    pred_phys = []

    for i in range(13, 25):
        column_index = 0
        if i in [15, 17, 21]:
            column_index = 1

        with open("../../data/"+str(userId)+"_"+str(i)+"up.txt") as f:
            lines = f.read().splitlines()
            touches = map(lambda x: x.split('\t'), lines[1:])

        touches = np.array(touches)
        locations = []
        req_string = ''
        bods = []
        
        for touch in touches:
            req_string = req_string + ''.join(touch[column_index]) 
            bod = dataIO.createlist(touch[column_index+4])
            
            if dataIO.contains_spikes(bod):
                continue
            
            bods.append(bod)
            locations.append(ast.literal_eval(touch[column_index+2]))
            
        bods = np.array(bods)
        locations=np.array(locations)

        for sentence in pool:
            if len(req_string)==0:
                break

            index = req_string.find(sentence)
            if index!=-1:
                req_sentences.append(sentence)
                typed_locations = np.array(locations[index:index+len(sentence)])
                typed_bod = np.array(bods[index:index+len(sentence)])

                req_string = req_string[:index]+req_string[index+len(sentence):]
                locations = np.delete(locations, np.s_[index:index+len(sentence)], 0)
                bods = np.delete(bods, np.s_[index:index+len(sentence)], 0)

                bod_data = scaler.transform(typed_bod)
                #pred = clf.predict(bod_data)
                pred = gp.predict(bod_data)
                #pred = np.argmax(pred, axis=1)
                vectors = np.concatenate((typed_locations, typed_locations **2),1)
                
                pred_x = np.zeros(len(vectors))
                pred_y = np.zeros(len(vectors))
                for i in range(len(vectors)):
                    #regr_no = pred[i]
                    for j in range(5):
                        pred_x[i] += pred[i][j]*regr_x[j].predict(vectors[i])
                        pred_y[i] += pred[i][j]*regr_y[j].predict(vectors[i])

                new_points = typed_locations + np.dstack((pred_x, pred_y))[0]
                
                typed_phys.append(dataIO.typed_phys(typed_locations))
                pred_phys.append(dataIO.typed_phys(new_points))

    phys_typed = []#np.zeros(len(req_sentences))
    phys_pred = []#np.zeros(len(req_sentences))
    
    #print typed_phys
    #print pred_phys
    for i in range(len(req_sentences)):    
        req_sentence = req_sentences[i]

        #phys_typed[i] = error_rate(req_sentence, typed_phys[i])
        #phys_pred[i] = error_rate(req_sentence, pred_phys[i])
        phys_typed.append(error_rate(req_sentence, typed_phys[i]))
        phys_pred.append(error_rate(req_sentence, pred_phys[i]))
        
    return phys_typed, phys_pred #mean


def run_old(userId):
    keys = postures.keys()
    locations = []
    bod = []
    targets_x = []
    targets_y = []
    y = []

    posture = 0
    for key in keys:
        filenos = postures[key]
        if key == "two_hand":
            a, b, c, d, e, f = dataIO.process_twohand(userId, posture)
            posture += 2
        else:
            a, b, c, d, e, f = dataIO.process_posture(userId, filenos, posture)
            posture += 1

        locations += a
        bod += b
        targets_x += c 
        targets_y += d 
        y += e

    locations = np.array(locations)
    bod = np.array(bod)
    targets_x = np.array(targets_x)
    targets_y = np.array(targets_y)
    y = np.array(y)

    locations, bod, targets_x, targets_y, y = utils.shuffle(locations, bod, targets_x, targets_y, y)
    locations = np.concatenate((locations, locations**2),1)

    scaler = preprocessing.StandardScaler().fit(bod)  
    bod_scaled = scaler.transform(bod)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1 ,1e-2, 1e-3],
                         'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1 ,1, 10, 100, 1000]}]

    clf = GridSearchCV(svm.SVC(C=1, cache_size=500), tuned_parameters, n_jobs=-1)
    #clf = svm.SVC(C=100, kernel='rbf', gamma=0.01, cache_size=500)
    clf.fit(bod_scaled, y)

    regr_x = [] 
    regr_y = []

    for i in range(0,5):
        index = np.where(y==i)[0]
        regr_x.append(learn_offset(locations[index], targets_x[index]))
        regr_y.append(learn_offset(locations[index], targets_y[index]))

    with open("../../Loggingapp/dataset.txt") as f:
        pool = f.read().splitlines()

    pool = sorted(pool, key=len, reverse=True)

    req_sentences = []
    typed_phys = []
    pred_phys = []

    for i in range(13, 25):
        column_index = 0
        if i in [15, 17, 21]:
            column_index = 1

        with open("../../data/"+str(userId)+"_"+str(i)+"up.txt") as f:
            lines = f.read().splitlines()
            touches = map(lambda x: x.split('\t'), lines[1:])

        touches = np.array(touches)
        req_string = ''.join(touches[:,column_index]) 
        locations = np.array(map(lambda x: ast.literal_eval(x), touches[:,(column_index+2)]))
        bod = np.array(map(lambda x: dataIO.createlist(x), touches[:,(column_index+4)]))

        for sentence in pool:
            if len(req_string)==0:
                break

            index = req_string.find(sentence)
            if index!=-1:
                req_sentences.append(sentence)
                typed_locations = np.array(locations[index:index+len(sentence)])
                typed_bod = np.array(bod[index:index+len(sentence)])

                req_string = req_string[:index]+req_string[index+len(sentence):]
                locations = np.delete(locations, np.s_[index:index+len(sentence)], 0)
                bod = np.delete(bod, np.s_[index:index+len(sentence)], 0)

                bod_data = scaler.transform(typed_bod)
                pred = clf.predict(bod_data)
                vectors = np.concatenate((typed_locations, typed_locations **2),1)

                pred_x = np.zeros(len(sentence))
                pred_y = np.zeros(len(sentence))
                for i in range(len(sentence)):
                    regr_no = pred[i]
                    pred_x[i] = regr_x[regr_no].predict(vectors[i])
                    pred_y[i] = regr_y[regr_no].predict(vectors[i])

                new_points = typed_locations + np.dstack((pred_x, pred_y))[0]

                typed_phys.append(dataIO.typed_phys(typed_locations))
                pred_phys.append(dataIO.typed_phys(new_points))

    phys_typed = []#np.zeros(len(req_sentences))
    phys_pred = []#np.zeros(len(req_sentences))
    for i in range(len(req_sentences)):    
        req_sentence = req_sentences[i]

        #phys_typed[i] = error_rate(req_sentence, typed_phys[i])
        #phys_pred[i] = error_rate(req_sentence, pred_phys[i])
        phys_typed.append(error_rate(req_sentence, typed_phys[i]))
        phys_pred.append(error_rate(req_sentence, pred_phys[i]))
        
    return phys_typed, phys_pred #mean
    

#typed= []
#pred = []
#for i in range(3, 18):
#    print "user "+ str(i)
#    a, b = run(i)
#    typed.append(a)    
#    pred.append(b)
    
#numpy.set_printoptions(precision=2)    
#print np.array(typed).flatten()
#print np.array(pred).flatten
#print    
#print np.mean(np.array(typed))
#print np.mean(np.array(pred))   
   
 
#[23.07, 17.14, 26.19, 11.41, 4.26, 13.19, 19.93, 13.79, 5.41, 18.74, 18.89, 19.11, 16.55, 18.16, 20.91]
#[23.04, 12.60, 27.05, 12.71, 7.94, 7.66, 25.01, 8.50, 5.94, 18.92, 14.41, 15.78, 14.83, 17.81, 16.82]

#16.45
#15.27

#16.2651977188
#14.9509547884




    
    