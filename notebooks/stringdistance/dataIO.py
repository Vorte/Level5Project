import ast, math

def within_button(touch, center):
    if center == (810.5, 245): # SPACE key
        if abs(touch[0]-center[0])>73/2.0 or abs(touch[1]-center[1])>250/2.0:
            return False
        return True
    
    if abs(touch[0]-center[0])>73/2.0 or abs(touch[1]-center[1])>43/2.0:
        return False
    return True

'''
def typed_string(touches):
    centers = get_key_centers()
    keys = centers.keys()
    typed_string = []
    
    for touch in touches:
        for key in keys:
            if within_button(touch, centers[key]):
            #if within_distance(touch, centers[key], 3):
                if key == 'SPACE':
                    typed_string.append(' ')
                else:
                    typed_string.append(key)
                break
        
    return ''.join(typed_string)
'''

def typed_string(touches):
    centers = get_key_centers()
    keys = centers.keys()
    typed_strings = [[],[],[],[],[],[],[]]
    
    for touch in touches:
        for key in keys:
            for i in range(1, 8):
                if within_distance(touch, centers[key], i):
                    if key == 'SPACE':
                        typed_strings[i-1].append(' ')
                    else:
                        typed_strings[i-1].append(key)
                    #continue
    
    for i in range(len(typed_strings)):
        typed_strings[i] = ''.join(typed_strings[i])
                    
    return typed_strings


def get_key_centers():
    data = {}
    
    filename = "/home/dimitar/Desktop/Python/Level5Project/Loggingapp/keylocations.txt"
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = map(lambda x: x.split(' '), lines)
        for item in lines:
            data[item[0]] = (float(item[1]), float(item[2]))
            
    return data

def within_distance(point1, point2, distance):
    dot_pitch = 0.101195219 # 0.1011
    dist_px = int(distance/dot_pitch)
    
    dist = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    if dist <= dist_px:
        return True
    return False


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

def createlist(string):
    return map(float, string.replace('(', '').replace(')', '').split(','))

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
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"up.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            for line in lines:
                letter = line[1]
                location = list(ast.literal_eval(line[3]))
                center = centers[letter]
                
#                if not iscorrect(location, center):
#                    continue
                
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
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"up.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            for line in lines:
                letter = line[0]
                location = list(ast.literal_eval(line[2]))
                center = centers[letter]
                
#                if not iscorrect(location, center):
#                    continue
                
                targets_x.append(center[0]-location[0])
                targets_y.append(center[1]-location[1])                
                touch_centers.append(center)
                locations.append(location)
                bod.append(createlist(line[-1]))
                y.append(posture)
                
    return locations, bod, targets_x, targets_y, y, touch_centers
    
    
    
