
postures = {"left_hand":["4", "8", "11"], "right_hand":["1", "7", "10"], 
            "index_finger":["3", "5", "12"], "two_hand":["2", "6", "9"]}


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
                  
def createlist(x):
    return map(float, x.replace('(', '').replace(')', '').split(','))
    
def read_twothumb(userid, session = -1):
    left = []
    right = []
    filenos = postures["two_hand"]
    
    if session>=0:
        filenos = filenos[session]
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: x.split('\t'), lines[1:])
            map(lambda x: left.append(createlist(x[-1])) 
                if x[0]=="left" else right.append(createlist(x[-1])), lines)
        
    return left, right
    

def read_file(userid, posture):
    data = []
    filenos = postures[posture]
    
    for fileno in filenos:
        filename = "/home/dimitar/Desktop/Python/experiment/results/"+str(userid)+"_"+fileno+"down.txt"
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            lines = map(lambda x: (x.split('\t')[-1]).replace('(', '').replace(')', ''), lines[1:])
            map(lambda x: data.append(map(float,x.split(', '))), lines)
        
    return data
    
