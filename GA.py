#----------------------------Imports------------------------------
import numpy as np
import pandas as pd
from meep.materials import Au
from func import *

#----------------------------Inputs------------------------------
vari = ['CytopThickness','GapThickness','SpacerRI','GoldThickness','Cr Thickness','Fitness']
deltaP = [0.005,0.005,0.01,0.005,0.001]
limit_min = [0.005,0.01,1.33,0.005,0.003]
limit_max = [0.300,0.5,1.34,0.06,0.015]
material = [3,0,4,6,0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
generation = 30
population = 6 #minimum of 4
step = 10
starter=0 
file = 'starter.txt'
resolution = 200       #pixels/um

#----------------------------Function------------------------------
ideal_peak = 0.647
weights = [0.33,0.33,0.33] #Peak Position, FWHM, Enhancement (sum to 1)

#----------------------------Variables------------------------------
Evolution = pd.DataFrame(columns=vari)
P_space = [(np.arange(limit_min[0],limit_max[0],deltaP[0])),(np.arange(limit_min[1],limit_max[1],deltaP[1])),(np.arange(limit_min[2],limit_max[2],deltaP[2])),(np.arange(limit_min[3],limit_max[3],deltaP[3])),(np.arange(limit_min[4],limit_max[4],deltaP[4]))]
f_cen,df,fmin,fmax,nfreq,sy,dpml,air,sx,thick,g,d2 = setvar()


init_refl_data,init_tran_flux = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,0,0,0,0,thick)
param = np.zeros((population,len(deltaP)))
F = np.zeros(population)
for j in range(population):
    spacer,d1,THICKNESS,P = paramset(P_space)
    geometry = geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml)
    en,Rs,wl = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,1,THICKNESS)
    F[j],output = fitness(weights,en,ideal_peak,Rs,wl)
    param[j,:] = P
Evolution = scribe(Evolution,param,F,vari)
#eve = output 
#migrate1 = param[np.argmax(F)]
par = roulette(F,P_space,param,0,generation)
pop = genes(par,P_space,population)
for i in range(generation):
    for j in range(population):
        cytop,spacer,d1,gold,cr = pop[j,:]
        THICKNESS = thicker(cytop,gold,cr)
        geometry = geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml)
        en,Rs,wl = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,1,THICKNESS)
        F[j],output = fitness(weights,en,ideal_peak,Rs,wl)
        param[j,:] = pop[j,:]
    Evolution = scribe(Evolution,param,F,vari)
    par = roulette(F,P_space,param,i,generation)
    pop = genes(par,P_space,population)
     
    
    #Evolutionary progress
    #Evolve changes the fitness parameters based on how well the substrate is doing
    #Migrate means that the substrate with the highest fitness has been there too long, and is moved into a new random population
    test = i/step
    if test.is_integer() == True:
        #migrate2=param[np.argmax(F)]
        #pop = migrate(migrate1,migrate2,pop,P_space)
        #migrate1 = migrate2
        #eve1 = output
        #weights = evolve(weights,output,eve,eve1) 
        #eve=eve1
        Evolution.to_csv('Evolution.csv',index=False)
        print(i) 
        
resolution=500
finalize(Evolution,material,resolution,init_refl_data,init_tran_flux,weights,ideal_peak)

      
    

