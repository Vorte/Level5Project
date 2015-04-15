
postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}


def read_twothumb_se(userid):
    left = []
    right = []
    left_test = []
    right_test= []
           
    filenos = copy.copy(postures["two_hand"])
    random.shuffle(filenos)
    
    test_file = filenos.pop()
    print "Session used for testing: "+test_file
    print
    
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
                  
def createlist(x):
    return map(float, x.replace('(', '').replace(')', '').split(','))

def contains_spikes(values):
    for value in values:
        if value>40000:
            return True
    
    return False
    
def read_twothumb(userid, session = -1):
    left = []
    right = []
    filenos = postures["two_hand"]
    
    if session>=0:
        filenos = filenos[session]
    
    for fileno in filenos:
        filename = "../../data/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            for line in lines:
                bod = createlist(line[-1])
                if contains_spikes(bod):
                    continue
                
                if line[0] == "left":
                    left.append(bod)
                else:
                    right.append(bod)
        
    return left, right
    

def read_file(userid, posture):
    data = []
    filenos = postures[posture]
    
    for fileno in filenos:
        filename = "../../data/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            for line in lines:
                bod = createlist(line[-1])
                if contains_spikes(bod):
                    continue
                
                data.append(bod)            
        
    return data











    
