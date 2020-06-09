#----------------------------Imports------------------------------
import numpy as np
import pandas as pd
from func import *

# Parameter space
vari = ['CytopThickness','GapThickness','SpacerRI','GoldThickness','Binder Thickness','Cr(0) or Ti(1)','Mirror(0) or no (1)','Fitness']
deltaP = [0.005,0.005,0.01,0.005,0.001,1,1]
limit_min = [0.005,0.01,1.33,0.005,0.003,0,0]
limit_max = [0.500,0.85,1.34,0.06,0.015,1.1,1.1]
# If the array is not evenly spaced, use the append command in line 28
weights = [0.33,0.33,0.33] #low reflection at ideal peak, large enhancement field, large enhancement
ideal_peak = [0.647]
savestart(deltaP,limit_max,limit_min,weights,ideal_peak)

# Summarization Documents
Evolution = pd.DataFrame(columns=vari)
Evolution.to_csv('Evolution.csv',index=False)
Progress = pd.DataFrame(columns=vari)
Progress.to_csv('Progress.csv',index=False)

# Set first population
population = 49
P_space = []
for i in range(len(deltaP)):
    P_space.append((np.arange(limit_min[i],limit_max[i],deltaP[i])))
#P_space.append(list_example)
param = np.zeros((population,len(P_space)))
for j in range(population):
    spacer,d1,THICKNESS,thick,P = paramset(P_space)
    param[j,:] = P
np.savetxt("population.txt",param)
