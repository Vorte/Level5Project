
def meetscond(string):
    chars = set('\'\".,')
    if len(string)<10 or len(string)>100 or any((c in chars) for c in string):
        return False
    return True

lines = []
with open("mobile_nvp.txt") as f:
    lines = f.read().splitlines()
    lines = map(lambda x: x.strip().split('\t')[1], lines)
    lines = filter(meetscond, lines)
   
with open("dataset.txt", "w") as f:
    for line in lines:
        f.write(line+"\n")    

