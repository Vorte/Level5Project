import numpy as np

postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}

def read_twothumb(userid):
    left = []
    right = []
    filenos = postures[twohand]
    createlist = lambda x: map(float, x.replace('(', '').replace(')', '').split(','))
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            map(lambda x: left.append(createlist(x[-1])) 
                if x[0]=="left" else right.append(createlist(x[-1])), lines)
        
    return np.array(left), np.array(right)
    

def read_file(userid, posture):
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
    
    
