import meep as mp
import numpy as np
from meep.materials import Au,Al,Ag,Cr,Ti
import random
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import seaborn as sb
import os
import pandas as pd


def setvar():
    g = 1.45
    d2 = 1.33
    dpml = 0.2                  #um
    air = 0.4                   #um
    THICKNESS = [0.400,0.050,0.700]             #um
    lambda_min = 0.55
    lambda_max = 0.75           # maximum source waveWIDTH
    thick=np.sum(THICKNESS)
    nfreq=60
    fmin = 1/lambda_max         # minimum source frequency
    fmax = 1/lambda_min         # maximum source frequency
    f_cen = 0.5*(fmin+fmax)     # source frequency center
    df = fmax-fmin              # source frequency width
    sx = 0.5
    sy = dpml+thick+air+dpml
    return f_cen,df,fmin,fmax,nfreq,sy,dpml,air,sx,thick,g,d2
    

def paramset(P_space):
    P = np.empty(len(P_space))
    for i in range(len(P_space)):
        P[i]=np.random.choice(P_space[i])
    cytop,spacer,d1,gold,cr = P#EDIT IF ADDING PARAMETERS
    THICKNESS = thicker(cytop,gold,cr)
    return spacer,d1,THICKNESS,P
    

def arresults(best_grating,file):
    if os.path.exists(file):
        with open(file, 'a') as filer :
            filer.write('\n'+repr(best_grating))
    else:
        filedata = best_grating
        np.savetxt(file,filedata)
    return

def simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,n,THICKNESS):
    #----------------------Simulation------------------------------
    cell = mp.Vector3(sx, sy)
    thick = np.sum(THICKNESS)
    #define Gaussian plane wave
    sources = [mp.Source(mp.GaussianSource(f_cen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(0,0.5*sy-dpml-0.02*air,0 ),
            size=mp.Vector3(x=sx))]
    #define pml layers
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Y, side=mp.High),mp.Absorber(thickness=dpml, direction=mp.Y, side=mp.Low)]
    tran_fr = mp.FluxRegion(center=mp.Vector3(0,-0.5*sy+dpml+0.05), size=mp.Vector3(x=sx) )
    refl_fr = mp.FluxRegion(center=mp.Vector3(0,0.5*sy-dpml-0.1*air), size=mp.Vector3(x=sx)) 
    mp.quiet(quietval=True)
    if n ==0:
        if os.path.exists('dft_Z_empty.h5'):
            os.remove('dft_Z_empty.h5')
        sim = mp.Simulation(cell_size=cell,
                boundary_layers=pml_layers,
                sources=sources,
                symmetries=[mp.Mirror(mp.X)],
                dimensions=2,
                resolution=resolution,
                k_point=mp.Vector3())
        
        #----------------------Monitors------------------------------
        refl = sim.add_flux(f_cen, df, nfreq, refl_fr)
        tran = sim.add_flux(f_cen,df, nfreq, tran_fr)                        
        dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,-sy*0.5+dpml+THICKNESS*0.5,0), size=mp.Vector3(sx,THICKNESS)))
        #----------------------Run------------------------------
        sim.run(until_after_sources=mp.stop_when_fields_decayed(5,mp.Ez,mp.Vector3(),1e-3))
        
        #----------------------Genetic------------------------------
        sim.output_dft(dfts_Z, "dft_Z_empty")
        init_refl_data = sim.get_flux_data(refl)
        init_tran_flux = mp.get_fluxes(tran)
        sim.reset_meep()
        get = init_refl_data,init_tran_flux
        
    elif n==1:
        if os.path.exists('dft_Z_fields.h5'):
            os.remove('dft_Z_fields.h5')
        sim = mp.Simulation(cell_size=cell,
                boundary_layers=pml_layers,
                sources=sources,
                geometry=geometry,
                symmetries=[mp.Mirror(mp.X)],
                dimensions=2,
                resolution=resolution,
                k_point=mp.Vector3())
        
        refl = sim.add_flux(f_cen, df, nfreq, refl_fr)
        tran = sim.add_flux(f_cen,df, nfreq, tran_fr)
        sim.load_minus_flux_data(refl,init_refl_data)
        dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,-sy*0.5+dpml+thick*0.5,0), size=mp.Vector3(sx,thick)))        
        sim.run(until_after_sources=mp.stop_when_fields_decayed(5,mp.Ez,mp.Vector3(),1e-3)) #mp.at_beginning(mp.output_epsilon),
        sim.output_dft(dfts_Z, "dft_Z_fields")
        flux_freqs = mp.get_flux_freqs(refl)
        final_refl_flux = mp.get_fluxes(refl)
        wl = []
        Rs = []
        for i in range(nfreq):
            wl = np.append(wl, 1/flux_freqs[i])
            Rs = np.append(Rs,-final_refl_flux[i]/init_tran_flux[i])
        sim.reset_meep()
        #get = np.min(Rs) 
        en = enhance(Rs,0)
        get=en,Rs,wl
    elif n==2:
        if os.path.exists('dft_Z_fields.h5'):
            os.remove('dft_Z_fields.h5')
        sim = mp.Simulation(cell_size=cell,
                boundary_layers=pml_layers,
                sources=sources,
                geometry=geometry,
		symmetries=[mp.Mirror(mp.X)],
                dimensions=2,
                resolution=resolution,
                k_point=mp.Vector3())
        
        refl = sim.add_flux(f_cen, df, nfreq, refl_fr)
        tran = sim.add_flux(f_cen,df, nfreq, tran_fr)
        sim.load_minus_flux_data(refl,init_refl_data)
        dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,-sy*0.5+dpml+thick*0.5,0), size=mp.Vector3(sx,thick)))
        sim.run(mp.at_beginning(mp.output_epsilon),until_after_sources=mp.stop_when_fields_decayed(5,mp.Ez,mp.Vector3(),1e-3)) #mp.at_beginning(mp.output_epsilon),
        sim.output_dft(dfts_Z, "dft_Z_fields")
        flux_freqs = mp.get_flux_freqs(refl)
        final_refl_flux = mp.get_fluxes(refl)
        final_tran_flux = mp.get_fluxes(tran)
        wl = []
        Rs = []
        Ts = []
        for i in range(nfreq):
            wl = np.append(wl, 1/flux_freqs[i])
            Rs = np.append(Rs,-final_refl_flux[i]/init_tran_flux[i])
            Ts = np.append(Ts,final_tran_flux[i]/init_tran_flux[i])
        As = 1-Rs-Ts
        plt.clf()
        plt.figure()
        plt.plot(wl,Rs,'bo-',label='reflectance')
        plt.plot(wl,Ts,'ro-',label='transmittance')
        plt.plot(wl,As,'go-',label='absorption')
        plt.xlabel("wavelength (μm)")
        plt.legend(loc="upper right")
        plt.savefig('Extinction.png')
        #get = np.min(Rs) 
        en = enhance(Rs,1)
        get=en,Rs,wl
        
        #plt.figure()
        #plt.plot(wl,Rs,'bo-',label='reflectance')
        eps = h5py.File('GA-eps-000000000.h5', 'r')
        eps = eps.get('eps').value
        Enhance = np.rot90(eps)
        plt.clf()
        plt.figure()
        heat_map = sb.heatmap(Enhance,cmap='plasma',xticklabels=False, yticklabels=False)
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.savefig('Epsilon.png')
        
            
    return(get)

def geom(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml):
    Glass = mp.Medium(epsilon=g)
    diel1 = mp.Medium(epsilon=d1)
    diel2 = mp.Medium(epsilon=d2)
    mat = [Au,Al,Ag,Glass,diel1,diel2,Cr]
    #----------------------------Geometry------------------------------
    substrate = mp.Block(mp.Vector3(mp.inf, THICKNESS[0],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[0]*0.5+dpml,0),  material=mat[material[0]])
    mirror = mp.Block(mp.Vector3(mp.inf, THICKNESS[1],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[1]*0.5+THICKNESS[0]+dpml,0),  material=mat[material[1]])
    cytop = mp.Block(mp.Vector3(mp.inf,THICKNESS[2],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[2]*0.5+THICKNESS[0]+THICKNESS[1]+dpml,0),  material=mat[material[2]])
    gold = mp.Block(mp.Vector3(mp.inf,THICKNESS[3],mp.inf), center=mp.Vector3(0, -sy*0.5+THICKNESS[3]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+dpml,0),  material=mat[material[3]])
    water = mp.Block(mp.Vector3(mp.inf, THICKNESS[4],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[4]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+THICKNESS[3]+dpml,0),  material=mat[material[4]])
    spacer =  mp.Block(mp.Vector3(spacer,THICKNESS[3],mp.inf), center=mp.Vector3(0, -sy*0.5+THICKNESS[3]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+dpml,0),  material=mat[material[4]])
    geometry = [substrate,mirror,cytop,gold,water,spacer]
    return(geometry)

def geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml):
    Glass = mp.Medium(epsilon=g)
    diel1 = mp.Medium(epsilon=d1)
    diel2 = mp.Medium(epsilon=d2)
    mat = [Au,Al,Ag,Glass,diel1,diel2,Cr]
    geometry=[]
    t=-sy*0.5+dpml
    for i in range(len(THICKNESS)):
        t+=THICKNESS[i]*0.5
        geometry.append(mp.Block(mp.Vector3(mp.inf, THICKNESS[i],mp.inf), center=mp.Vector3(0,t,0),  material=mat[material[i]]))
        t+=THICKNESS[i]*0.5
    geometry.append(mp.Block(mp.Vector3(spacer,THICKNESS[-3]+THICKNESS[-2],mp.inf), center=mp.Vector3(0, -(THICKNESS[-2]+THICKNESS[-3])*0.5+t-THICKNESS[-1],0),  material=mat[material[-1]]))
    return(geometry)
    
def roulette(F,P_space,P):
    F = (F-np.min(F))/(np.max(F)-np.min(F))
    idx = np.argsort(F)  
    F = np.array(F)[idx]
    pop = np.array(P)[idx]
    sizer = len(P[0])
    parent=np.zeros((3,sizer))
    parent[2,:]=pop[-1,:]
    x=0
    NOPE=len(F)+1
    for j in range(2):
        doh = random.uniform(0,1)
        for i in range(len(F)):
            if i == NOPE:
                pass
            else:
                if doh <= F[i]:
                    NOPE = i
                    parent[x,:]=pop[i,:]
                    x+=1
                    doh = 2
    return(parent)
    
def genes(par,P_space,population):
    sizer = len(par[0])
    pop = np.zeros((population,sizer))
    x=paramset(P_space)
    pop[3,:]=x[3]
    for i in range(sizer):
        doh = random.uniform(0,1)
        if doh < 0.2:
            pop[1,i]=np.random.choice(P_space[i])
        else:
            pop[1,i] = par[2,i]
    for i in range(sizer):
        doh = random.uniform(0,1)
        if doh < 0.5:
            pop[2,i] = np.random.choice(P_space[i])
        else:
            pop[2,i] = par[2,i]
    for i in range(sizer):
        pop[0,i] = par[2,i]
    if population > 4:
        for j in range(population-4):
           doh = random.randint(0,sizer)
           pop[j+4,doh:] = par[1,doh:]
           pop[j+4,:doh] = par[2,:doh]
    return(pop)

def finalize(Evolution,material,resolution,init_refl_data,init_tran_flux,weights,ideal_peak):
    eve = Evolution.to_numpy()
    f_cen,df,fmin,fmax,nfreq,sy,dpml,air,sx,thick,g,d2 = setvar()
    init_refl_data,init_tran_flux = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,0,0,0,0,thick)
    cytop,spacer,d1,gold,cr,fit = eve[-1,:] #EDIT IF ADDING PARAMETERS
    THICKNESS = thicker(cytop,gold,cr)
    geometry = geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml)
    en,Rs,wl = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,2,THICKNESS)
    F,output = fitness(weights,en,ideal_peak,Rs,wl)
    counts = np.arange(0,len(eve))
    fit = eve[:,-1]
    plt.clf()
    plt.figure()
    plt.plot(counts,fit,'.')
    plt.xlabel("counts")
    plt.ylabel("fitness")
    plt.savefig('Progress.png')
    print(output)
    Evolution.to_csv('Evolution.csv',index=False)
    return()

def thicker(cytop,gold,cr):
    THICKNESS = [0.400-cytop-cr,0.050,cytop,cr,gold,0.700-gold]
    return(THICKNESS)
    
def posty(saint,best_grating,progress,col,row,file):
    pic = np.zeros([row*100,col*100],dtype=np.uint8)
    x=0
    y=100
    z1=0
    z2=100
    for j in range(row):
        for i in range(col):
            pic[z1:z2,x:y] = (saint[i]/np.max(saint))* 255 
            y+=100
            x+=100
        z1+=100
        z2+=100
    img = Image.fromarray(pic)
    img.save('LOOK.png')
    #os.remove('dft_Z_fields.h5')
    #os.remove('dft_Z_empty.h5')    
    #arresults(best_grating,file)
    gen = np.arange(0,len(progress))
    plt.figure()
    plt.plot(gen,progress,'bo')
    plt.xlabel("Iteration Number")
    plt.ylabel("Max. Enhancement")
    plt.savefig('Progress.png')
    return()

def enhance(R,n):
    Es = h5py.File('dft_Z_fields.h5', 'r')
    Eo = h5py.File('dft_Z_empty.h5', 'r')
    
    ei2 = Es.get('ez_'+repr(np.argmin(R))+'.i').value     
    er2 = Es.get('ez_'+repr(np.argmin(R))+'.r').value 
    E2 = ei2**2+er2**2
    ei1 = Eo.get('ez_'+repr(np.argmin(R))+'.i').value     
    er1 = Eo.get('ez_'+repr(np.argmin(R))+'.r').value 
    E1 = ei1**2+er1**2
    Enhance = E2/E1
    Enhance = np.rot90(Enhance)
    if n == 1:
        plt.clf()
        plt.figure()
        heat_map = sb.heatmap(Enhance,cmap='plasma',xticklabels=False, yticklabels=False)
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.savefig('Enhance_Z.png')
    return(np.max(np.max(Enhance)))    

def imagine(saint,par,sizer):
    for i in range(sizer):
        saint[i]=saint[i]+par[2,i]
    return(saint)

    
def fitness(weights,en,ideal_peak,Rs,wl):
    dl = 1/((np.abs(wl[np.argmin(Rs)]-ideal_peak))*1000)
    f_half = Rs - np.min(Rs)/2
    maxi = np.argmin(Rs)
    if len(f_half[:maxi]) == 0:
        fwhm = 0
    elif len(f_half[maxi:]) == 0:
        fwhm = 0
    else:
        fwhm = 1/((wl[np.argmin(f_half[:maxi])]*1000) - (wl[maxi+np.argmin(f_half[maxi:])]*1000))
    en=en/20
    output = [dl,fwhm,en]
    F = np.dot(weights,output)
    return F,output

def evolve(weights,output,eve,eve1):
    delta = []
    zip_object = zip(eve, eve1)
    for eve, eve1 in zip_object:
        delta.append(eve1-eve)
    return(weights)

def migrate(migrate1,migrate2,pop,P_space):
    x = (migrate1 == migrate2)
    if np.sum(x)==len(x):
        for i in range(len(pop)-1):
            x = paramset(P_space)
            pop[i+1,:] = x[3]
    else:
        pass
    return(pop)

def scribe(Evolution,param,F,vari):
    bestie = param[np.argmax(F)]
    bestie = np.append(bestie,np.max(F))
    best = pd.DataFrame(bestie.reshape(-1, len(bestie)),columns=vari)
    Evolution=Evolution.append(best)
    return(Evolution)
        