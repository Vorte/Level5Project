import numpy as np
import random, copy, math, ast

'''
class Touch(object):
    def __init__(self, x, y, bod, letter, time, left = True, posture="two_hand"):
        self.x = x
        self.y = y
        self.bod = bod
        self.time = time
        self.letter = letter
        self.left = left # False = right
        self.posture = posture
'''

postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}
            
def createlist(string):
    return map(float, string.replace('(', '').replace(')', '').split(','))
            
'''
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
'''

def within_button(touch, center):
    if abs(a[0]-b[0])>43 or abs(a[0]-b[0])>73:
        return False
    return True
    
def typed_string(touches):
    centers = get_key_centers()
    keys = centers.keys()
    typed_string = []
    
    for touch in touches:
        for key in keys:
            if within_button(touch, centers[key]):
                typed_string.append(key)
                break
        
    return ''.join(typed_string)

def iscorrect(a, b):
    key_width = 43    
    dist = math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )
    
    if dist > 2*key_width:
        return False
    return True


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
    bod = []
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
                
                if not iscorrect(location, center):
                    continue
                
                if line[0] == "left":
                    y.append(0+posture)
                else:
                    y.append(1+posture)
                
                touch_centers.append(center)
                targets_x.append(center[0]-location[0])
                targets_y.append(center[1]-location[1])            
                locations.append(location)
                bod.append(createlist(line[-1]))
                
    return locations, bod, targets_x, targets_y, y, touch_centers


def process_posture(userid, filenos, posture): 
    
    locations = []
    bod = []
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
                
                if not iscorrect(location, center):
                    continue
                
                targets_x.append(center[0]-location[0])
                targets_y.append(center[1]-location[1])                
                touch_centers.append(center)
                locations.append(location)
                bod.append(createlist(line[-1]))
                y.append(posture)
                
    return locations, bod, targets_x, targets_y, y, touch_centers




    
    
    
