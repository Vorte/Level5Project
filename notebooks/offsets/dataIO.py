import numpy as np
import random, copy, math

class Touch(object):
    def __init__(self, x, y, bod, letter, time, left = True, posture="two_hand"):
        self.x = x
        self.y = y
        self.bod = bod
        self.time = time
        self.letter = letter
        self.left = left # False = right
        self.posture = posture

postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}
            
def createlist(string):
    return map(float, string.replace('(', '').replace(')', '').split(','))
            
def read_twothumb_se(userid): #BOD
    left = []
    right = []
    left_test = []
    right_test= []
           
    filenos = copy.copy(postures["two_hand"])
    random.shuffle(filenos)
    
    test_file = filenos.pop()
    print "Session used for testing: "+test_file
    print
    #createlist = lambda x: map(float, x.replace('(', '').replace(')', '').split(','))
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            map(lambda x: left.append(createlist(x[-1])) 
                if x[0]=="left" else right.append(createlist(x[-1])), lines)

    filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+test_file+"down.txt"
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = map(lambda x: x.split('\t'), lines[1:])
        map(lambda x: left_test.append(createlist(x[-1])) 
            if x[0]=="left" else right_test.append(createlist(x[-1])), lines)
            
    return np.array(left), np.array(right), np.array(left_test), np.array(right_test)
                  

def read_twothumb(userid): #BOD
    left = []
    right = []
    filenos = postures["two_hand"]
    createlist = lambda x: map(float, x.replace('(', '').replace(')', '').split(','))
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            map(lambda x: left.append(createlist(x[-1])) 
                if x[0]=="left" else right.append(createlist(x[-1])), lines)
        
    return np.array(left), np.array(right)
    

def read_file(userid, posture): #BOD
    # TODO: up/down switching
    data = []
    filenos = postures[posture]
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: (x.split('\t')[-1]).replace('(', '').replace(')', ''), lines[1:])
            map(lambda x: data.append(map(float,x.split(', '))), lines)
        
    return np.array(data) 
    

def get_touch_locations(userid, posture):
    data = {}
    filenos = postures[posture]
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"up.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            map(lambda x: x.split('\t'), lines)
            print lines
        
    return np.array(data)
    

def filter_new(touches):
    centers = get_key_centers()
    filtered = []
    
    count = 0
    for touch in touches:
        coord = (touch.x, touch.y)
        center = centers[touch.letter]        
        if iscorrect(coord, center):
            filtered.append(touch)
        else:
            count += 1
                    
    print ("Filtered %d points." %count)
                
    return filtered

def filter_touches(touches):
    centers = get_key_centers()
    keys = touches.keys()
    filtered = {}
    
    count = 0
    for key in keys:
        touch_coords = touches[key]
        center = centers[key]        
        for coord in touch_coords:
            if iscorrect(coord, center):
                filtered.setdefault(key,[]).append(coord)
            else:
                count += 1
                    
    print ("Filtered %d points." %count)
    print
                
    return filtered

def iscorrect(a, b):
    key_width = 43
#    if abs(a[0]-b[0])>43 or abs(a[0]-b[0])>73:
#        return False
    
    dist = math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )
    
    if dist > 2*key_width:
        return False
    return True


def get_key_centers():
    data = {}
    
    filename = "/home/dimitar/Desktop/Python/Level5Project/Loggingapp/keylocations.txt"
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = map(lambda x: x.split(' '), lines)
        for item in lines:
            data[item[0]] = (float(item[1]), float(item[2]))
            
    return data






    
    
    
