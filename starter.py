#----------------------------Imports------------------------------
import numpy as np
import pandas as pd
from func import *

# Parameter space
#period,grating_t,grating_w,sep,mthick,mat
vari = ['Period','Grating_T','Grating_W','Separation','Metal_Thickness','Au(0) or Ag(1)','Fitness']
deltaP = [0.010,0.010,0.010,0.005,0.002,1]
limit_min = [0.200,0.05,0.05,0.0,0.005,0]
limit_max = [0.700,0.4,0.200,1.0,0.050,1.1]
# If the array is not evenly spaced, use the append command in line 28
weights = [1] #low reflection at ideal peak, large enhancement field, large enhancement
ideal_peak = [0.570]
savestart(deltaP,limit_max,limit_min,weights,ideal_peak)

# Summarization Documents
Evolution = pd.DataFrame(columns=vari)
Evolution.to_csv('Evolution.csv',index=False)
Progress = pd.DataFrame(columns=vari)
Progress.to_csv('Progress.csv',index=False)

# Set first population
population = 15
P_space = []
for i in range(len(deltaP)):
    P_space.append((np.arange(limit_min[i],limit_max[i],deltaP[i])))
#P_space.append(list_example)
param = np.zeros((population,len(P_space)))
for j in range(population):
    param[j,:] = paramset(P_space)
np.savetxt("population.txt",param)