import predictpostures, predictposturesGP, averagepostures

noregr = []
pred = []
predGP = []
averGP = []

noregr_mse = []
pred_mse = []
predGP_mse = []
averGP_mse = []

for i in range(3, 18):
    print ("Processing user %d" %i)
    
    before, after, mse_bef, mse_aft = predictpostures.run(i)
    pred.append(after[1])
    pred_mse.append(mse_aft)
    noregr.append(before[1])
    noregr_mse.append(mse_bef)
    
    before, after, mse_bef, mse_aft = predictposturesGP.run(i)
    predGP.append(after[1])
    predGP_mse.append(mse_aft)
    
    before, after, mse_bef, mse_aft = averagepostures.run(i)
    averGP.append(after[1])
    averGP_mse.append(mse_aft)

print "###################"    
print "Processing Complete"  
print noregr
print pred 
print predGP 
print averGP 
print
print noregr_mse 
print pred_mse 
print predGP_mse 
print averGP_mse 

