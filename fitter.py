#----------------------------Imports------------------------------
import numpy as np
import pandas as pd
import os
from func import *

P_space=loadstart()
data = loadloads()
data = data[data != 0]
data = np.reshape(data, (len(data),-1))
Evolution = pd.read_csv("Evolution.csv")
vari = list(Evolution.columns.values.tolist())
Progress = pd.read_csv("Progress.csv")
param = np.loadtxt("population.txt")
population = len(param)
x = np.concatenate((param,data),axis=1)
x = pd.DataFrame(x,columns=vari)
Progress = pd.concat([x, Progress], ignore_index=True)
Progress.to_csv('Progress.csv',index=False)
Evolution = scribe(Evolution,param,vari,data)
Evolution.to_csv('Evolution.csv',index=False)
pop = genes(data,P_space,population,param)
np.savetxt("population.txt",pop)