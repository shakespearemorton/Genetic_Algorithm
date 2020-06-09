import meep as mp
import numpy as np
from meep.materials import Au,Al,Ag,Cr,Ti
import random
import matplotlib.pyplot as plt
import h5py
import seaborn as sb
import os
import pandas as pd

def genout(population):
    cytop,spacer,d1,gold,cr,ad,mir = population
    THICKNESS,thick = thicker(cytop,gold,cr,mir)
    if mir ==0:
        material = [3,0,4,int(6+ad),0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
    elif mir == 1:
        material = [3,int(6+ad),0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
    resolution = 400       #pixels/um
    f_cen,df,fmin,fmax,nfreq,sy,dpml,air,sx,g,d2 = setvar(thick)
    init_refl_data,init_tran_flux = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,0,0,0,0,thick)
    geometry = geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml)
    en,Rs,wl = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,1,THICKNESS)
    ideal_peak = np.loadtxt('ideal_peak.txt')
    weights = np.loadtxt('weights.txt')
    F = fitness(weights,en,ideal_peak,Rs,wl)
    return(F)

def loadloads():
    filenum = np.arange(0,50)
    data = np.zeros((len(filenum),1))
    for i in range(len(filenum)):
        file = str(repr(filenum[i])+'.txt')
        if os.path.exists(file):
            F = np.loadtxt(file)
            data[i,:] = F
    return(data)

def setvar(thick):
    g = 1.45
    d2 = 1.33
    dpml = 0.2                  #um
    air = 0.4                   #um
    lambda_min = 0.40
    lambda_max = 0.75           # maximum source waveWIDTH
    nfreq=100
    fmin = 1/lambda_max         # minimum source frequency
    fmax = 1/lambda_min         # maximum source frequency
    f_cen = 0.5*(fmin+fmax)     # source frequency center
    df = fmax-fmin              # source frequency width
    sx = 0.5
    sy = dpml+thick+air+dpml
    return f_cen,df,fmin,fmax,nfreq,sy,dpml,air,sx,g,d2
    

def paramset(P_space):
    P = np.empty(len(P_space))
    for i in range(len(P_space)):
        P[i]=np.random.choice(P_space[i])
    cytop,spacer,d1,gold,cr,ad,mir = P#EDIT IF ADDING PARAMETERS
    THICKNESS,thick = thicker(cytop,gold,cr,mir)
    return spacer,d1,THICKNESS,thick,P
    
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
        #if os.path.exists('dft_Z_empty.h5'):
            #os.remove('dft_Z_empty.h5')
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
        #if os.path.exists('dft_Z_fields.h5'):
            #os.remove('dft_Z_fields.h5')
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
        #if os.path.exists('dft_Z_fields.h5'):
            #os.remove('dft_Z_fields.h5')
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
        eps = h5py.File('_last-eps-000000000.h5', 'r')
        eps = eps.get('eps').value
        Enhance = np.rot90(eps)
        plt.clf()
        plt.figure()
        heat_map = sb.heatmap(Enhance,cmap='plasma',xticklabels=False, yticklabels=False)
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.savefig('Epsilon.png')
        
            
    return(get)

def geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml):
    Glass = mp.Medium(epsilon=g)
    diel1 = mp.Medium(epsilon=d1)
    diel2 = mp.Medium(epsilon=d2)
    mat = [Au,Al,Ag,Glass,diel1,diel2,Cr,Ti]
    geometry=[]
    t=-sy*0.5+dpml
    for i in range(len(THICKNESS)):
        if THICKNESS[i] == 0:
            pass
        else:
            t+=THICKNESS[i]*0.5
            geometry.append(mp.Block(mp.Vector3(mp.inf, THICKNESS[i],mp.inf), center=mp.Vector3(0,t,0),  material=mat[material[i]]))
            t+=THICKNESS[i]*0.5
    geometry.append(mp.Block(mp.Vector3(spacer,THICKNESS[-3]+THICKNESS[-2],mp.inf), center=mp.Vector3(0, -(THICKNESS[-2]+THICKNESS[-3])*0.5+t-THICKNESS[-1],0),  material=mat[material[-1]]))
    return(geometry)
    
def roulette(F,P_space,P,i,generation):
    F1 = (F-np.min(F))/(np.max(F)-np.min(F))
    idx = np.argsort(F1)  
    F1 = np.array(F1)[idx]
    pop = np.array(P)[idx]
    sizer = len(P[0])
    parent=np.zeros((3,sizer))
    parent[2,:]=pop[-1,:]
    if i > generation/2:
        pass
    else:
        temp = 100*(1-(i/(generation/2)))
        F = np.exp(F/temp)
    F = (F-np.min(F))/(np.max(F)-np.min(F))
    idx = np.argsort(F)  
    F = np.array(F)[idx]
    pop = np.array(P)[idx]
    x=0
    NOPE=len(F)+1
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

def sus(population,F1):
    Npointer = int(population/2)
    parents=[]
    addme = 1/Npointer
    pointer = random.uniform(0,addme)
    i=1
    k=0
    sume=0
    
    while k < Npointer:
        sume+=F1[-i]
        if sume >= pointer:
            parents.append(len(F1)-i)
            pointer+=addme
            k+=1
        i+=1
    random.shuffle(parents)
    return(parents)
    
    
def genes(F,P_space,population,param):
    #Arrange Fittest
    sizer=len(param[1])
    F1 = (F-np.min(F))/(np.max(F)-np.min(F))
    idx = np.argsort(F1,axis=0)  
    F1 = np.array(F1)[idx]
    F1 = np.reshape(F1, (len(F),-1))
    fit = np.array(param)[idx]
    fit = np.reshape(fit, (len(fit),-1))
    pop = np.zeros((np.shape(param)))
    F2 = np.array(F)[idx]
    F2 = F2/sum(F2)
    F2 = np.reshape(F2, (len(F),-1))
   
    #Distribution of Genetic Pool
    mut = int(0.13*population)
    offs = int(0.63*population)
    rand = population-1-mut-offs
    k = 0
    
    #Fittest Continues
    pop[k,:] = fit[-1,:]
    #Mutate
    k+=1
    for j in range(mut):
        for i in range(sizer):
            doh = random.uniform(0,1)
            if doh < 0.3:
                pop[k,i]=np.random.choice(P_space[i])
            else:
                pop[k,i] = pop[0,i]
        k+=1
    #Offspring
    parents = sus(population,F2)
    i=0
    for j in range(offs):
        doh = random.randint(0,sizer)
        pop[k,doh:] = fit[random.choice(parents),doh:]
        pop[k,:doh] = fit[random.choice(parents),:doh]
        k+=1
    #Random
    for j in range(rand):
        x=paramset(P_space)
        pop[k,:]=x[-1]
        k+=1
    return(pop)

def finalize(Evolution,resolution):
    eve = Evolution.to_numpy()
    
    cytop,spacer,d1,gold,cr,ad,mir,fit = eve[-1,:] #EDIT IF ADDING PARAMETERS
    if mir ==0:
        material = [3,0,4,int(6+ad),0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
    elif mir == 1:
        material = [3,int(6+ad),0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
    THICKNESS,thick = thicker(cytop,gold,cr,mir)
    f_cen,df,fmin,fmax,nfreq,sy,dpml,air,sx,g,d2 = setvar(thick)
    init_refl_data,init_tran_flux = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,0,0,0,0,thick)
    geometry = geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml)
    en,Rs,wl = simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,2,THICKNESS)
    counts = np.arange(0,len(eve))
    fit = eve[:,7]
    plt.clf()
    plt.figure()
    plt.plot(counts,fit,'.')
    plt.xlabel("counts")
    plt.ylabel("fitness")
    plt.savefig('Progress.png')
    return()

def thicker(cytop,gold,cr,mir):
    if mir == 0:
        THICKNESS = [1.000-cytop-cr,0.050,cytop,cr,gold,0.700-gold]
    elif mir ==1:
        THICKNESS = [1.000-cr,cr,gold,0.700-gold]
    thick = np.sum(THICKNESS)
    return(THICKNESS,thick)

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

def fitness(weights,en,ideal_peak,Rs,wl):
    dl = 1/(np.exp(np.abs(wl[np.argmin(Rs)]-ideal_peak)/ideal_peak)**5)
    f_half = Rs - np.min(Rs)/2
    maxi = np.argmin(Rs)
    if len(f_half[:maxi]) == 0:
        fwhm = 0
    elif len(f_half[maxi:]) == 0:
        fwhm = 0
    else:
        fwhm = ((wl[np.argmin(f_half[:maxi])]*1000) - (wl[maxi+np.argmin(f_half[maxi:])])*1000)
    fwhm=1/np.exp(np.abs(10-fwhm)/10)
    en=1/np.exp(np.abs(en-20)/20)
    output = [dl,fwhm,en]
    F = np.dot(weights,output)
    return F

def scribe(Evolution,param,vari,data):
    bestie = param[np.argmax(data)]
    bestie = np.append(bestie,np.max(data))
    best = pd.DataFrame(bestie.reshape(-1, len(bestie)),columns=vari)
    Evolution=Evolution.append(best)
    return(Evolution)

def loadstart():
    limit_max=np.loadtxt('limit_max.txt')
    limit_min=np.loadtxt('limit_min.txt')
    deltaP=np.loadtxt('deltaP.txt')
    P_space = []
    for i in range(len(deltaP)):
        P_space.append((np.arange(limit_min[i],limit_max[i],deltaP[i])))
    return(P_space)

def savestart(deltaP,limit_max,limit_min,weights,ideal_peak):
    limit_min = np.array(limit_min)
    limit_max = np.array(limit_max)
    deltaP = np.array(deltaP)
    weights = np.array(weights)
    ideal_peak = np.array(ideal_peak)
    np.savetxt('ideal_peak.txt',ideal_peak)
    np.savetxt('weights.txt',weights)
    np.savetxt('limit_min.txt',limit_min)
    np.savetxt('limit_max.txt',limit_max)
    np.savetxt('deltaP.txt',deltaP)
    return()
