#----------------------------Imports------------------------------
import numpy as np
from meep.materials import Au
from func import *

#----------------------------Inputs------------------------------
cytop = 0.100
col = 100  
row=4
d1 = 1.33
g = 1.45
d2 =1.34
mat_options=[0,5,2]
material = [3,0,4,0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5)]
side = 1
generation = 40
population = 8

#----------------------------Variables------------------------------
grate=0.005
dpml = 0.2                  #um
air = 0.4                   #um
resolution = 100         #pixels/um
THICKNESS = [0.100,0.050,cytop,0.010,0.500 ]             #um
lambda_min = 0.30
lambda_max = 0.800           # maximum source waveWIDTH
PADDING = lambda_max 
thick=np.sum(THICKNESS)
nfreq=60
fmin = 1/lambda_max         # minimum source frequency
fmax = 1/lambda_min         # maximum source frequency
f_cen = 0.5*(fmin+fmax)     # source frequency center
df = fmax-fmin              # source frequency width
sx = (col)*grate
sy = dpml+thick+air+dpml

init_refl_data,init_tran_flux = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,0,0,0,0,grate,THICKNESS)

progress=np.zeros(generation)
R=np.zeros(population)
Ref=np.zeros(population)
pop=np.zeros((population,col*row))
for j in range(population):
    matter = np.random.choice(mat_options, size=(col*row))
    pop[j,:]=matter
    geometry = geom(THICKNESS,sx,sy,grate,g,d1,d2,material,col,row,matter,dpml)
    en,re = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,1,grate,THICKNESS)
    R[j]=en
    Ref[j]=re
R_init=R
saint = np.zeros(col*row)
for i in range(generation):
    par = roulette(R,pop,col*row)
    pop2=pop
    saint = imagine(saint,par,col*row)
    pop = genes(par,population,col*row,mat_options)
    pop3=pop
    R=np.zeros(population)
    for j in range(population):
        matter=pop[j,:]
        geometry = geom(THICKNESS,sx,sy,grate,g,d1,d2,material,col,row,matter,dpml)
        en,re = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,1,grate,THICKNESS)
        R[j]=en
        Ref[j]=re
    print(i)
    progress[i]=np.max(R)
R_final=R    
best_grating = pop[np.argmin(R),:]       
saint = saint/generation    
print(progress)
print(best_grating)
print(Ref)
print(R_final)    
end = posty(saint,best_grating,progress,col)
      
    

