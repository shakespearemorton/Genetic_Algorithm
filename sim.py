import meep as mp
import numpy as np
from meep.materials import Au,Al,Ag,Cr,Ti
import random
import matplotlib.pyplot as plt
import h5py
import seaborn as sb
import os
import pandas as pd

def simulation(population):
    #----------------------Variables------------------------------
    cytop,spacer,d1,gold,cr,ad,mir = population
    if mir == 0:
        THICKNESS = [0.500,0.050,cytop,cr,gold,0.700]
        material = [3,0,4,int(6+ad),0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
    elif mir ==1:
        THICKNESS = [0.500,cr,gold,0.700]
        material = [3,int(6+ad),0,5] #[Au(0),Al(1),Ag(2),Glass(3),diel1(4),diel2(5),Cr(6)]
    thick = np.sum(THICKNESS)
    resolution = 100       #pixels/um
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
    sx = 1.0
    sy = dpml+thick+air+dpml
    enhance_surf_max = 0.050        #measure enhancement in the area
    enhance_surf_min = 0.010
    denhance = enhance_surf_max - enhance_surf_min
    
    #----------------------Simulation------------------------------
    cell = mp.Vector3(sx, sy)
    #define Gaussian plane wave
    sources = [mp.Source(mp.GaussianSource(f_cen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(0,0.5*sy-dpml-0.02*air,0 ),
            size=mp.Vector3(x=sx))]
    #define pml layers
    pml_layers = [mp.PML(thickness=dpml, direction=mp.Y, side=mp.High),mp.Absorber(thickness=dpml, direction=mp.Y, side=mp.Low)]
    tran_fr = mp.FluxRegion(center=mp.Vector3(0,-0.5*sy+dpml+0.05), size=mp.Vector3(x=sx) )
    refl_fr = mp.FluxRegion(center=mp.Vector3(0,0.5*sy-dpml-0.1*air), size=mp.Vector3(x=sx)) 
    #mp.quiet(quietval=True)
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
    dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,sy*0.5-dpml-air-THICKNESS[-1]+enhance_surf_min+0.5*denhance,0), size=mp.Vector3(sx,denhance)))
    #----------------------Run------------------------------
    sim.run(until_after_sources=mp.stop_when_fields_decayed(5,mp.Ez,mp.Vector3(),1e-3))
        
    #----------------------Reset------------------------------
    sim.output_dft(dfts_Z, "dft_Z_empty")
    init_refl_data = sim.get_flux_data(refl)
    init_tran_flux = mp.get_fluxes(tran)
    sim.reset_meep()
    get = init_refl_data,init_tran_flux
    print(THICKNESS)
    geometry = geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml)
    sim = mp.Simulation(cell_size=cell,
        boundary_layers=pml_layers,
        sources=sources,
        geometry = geometry,
        symmetries=[mp.Mirror(mp.X)],
        dimensions=2,
        resolution=resolution,
        k_point=mp.Vector3())    
        
    refl = sim.add_flux(f_cen, df, nfreq, refl_fr)
    tran = sim.add_flux(f_cen,df, nfreq, tran_fr)
    sim.load_minus_flux_data(refl,init_refl_data)
    dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,sy*0.5-dpml-air-THICKNESS[-1]+enhance_surf_min+0.5*denhance,0), size=mp.Vector3(sx,denhance)))        
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
    ideal = np.loadtxt('ideal_peak.txt') 
    wl = np.array(wl)
    close = np.abs(wl - ideal)
    dl = As[np.argmin(close)]
    Es = h5py.File('dft_Z_fields.h5', 'r')
    Eo = h5py.File('dft_Z_empty.h5', 'r')
    ei2 = Es.get('ez_'+repr(np.argmin(close))+'.i').value     
    er2 = Es.get('ez_'+repr(np.argmin(close))+'.r').value 
    E2 = ei2**2+er2**2
    ei1 = Eo.get('ez_'+repr(np.argmin(close))+'.i').value     
    er1 = Eo.get('ez_'+repr(np.argmin(close))+'.r').value 
    E1 = ei1**2+er1**2
    Enhance = E2/E1
    en = np.max(np.max(Enhance))
    enorm = (Enhance - np.min(np.min(Enhance)))/(np.max(np.max(Enhance))-np.min(np.min(Enhance)))
    en_avg = np.mean(enorm)
    en=1/np.exp(np.abs(en-20)/20)
    output = [dl,en_avg,en]
    weights = np.loadtxt('weights.txt')
    F = np.dot(weights,output)
    return(F)

def geom2(THICKNESS,sx,sy,g,d1,d2,material,spacer,dpml):
    Glass = mp.Medium(index=g)
    diel1 = mp.Medium(index=d1)
    diel2 = mp.Medium(index=d2)
    mat = [Au,Al,Ag,Glass,diel1,diel2,Cr,Ti]
    geometry=[]
    t=-sy*0.5+dpml
    for i in range(len(THICKNESS)):
        t+=THICKNESS[i]*0.5
        geometry.append(mp.Block(mp.Vector3(mp.inf, THICKNESS[i],mp.inf), center=mp.Vector3(0,t,0),  material=mat[material[i]]))
        t+=THICKNESS[i]*0.5
    geometry.append(mp.Block(mp.Vector3(spacer,THICKNESS[-3]+THICKNESS[-2],mp.inf), center=mp.Vector3(0, -(THICKNESS[-2]+THICKNESS[-3])*0.5+t-THICKNESS[-1],0),  material=mat[5]))
    return(geometry)

pop = np.loadtxt('population.txt')
for i in range(len(pop)):
    population = pop[i,:]
    F = simulation(population)
    fit = np.zeros(1)
    fit[0] = F
    np.savetxt(repr(i)+'.txt',fit)
