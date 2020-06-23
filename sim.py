import meep as mp
import numpy as np
from meep.materials import Ag, Au
import os
import h5py

def Simulation(period,grating_t,grating_w,sep,mthick,mat):
    
    #----------------------Variables------------------------------
    sub = 1.4                   #thickness of the substrate (where the grating is embedded)
    d1 = 1.52                 #dielectric constant of medium
    dpml = 0.2                  #um
    air = 1.5                   #um
    matty = [Au,Ag]
    #-------------------------Clear-------------------------------
    #if os.path.exists("dft_X_empty.h5"):
    #    os.remove("dft_X_empty.h5")
    #if os.path.exists("dft_X_fields.h5"):
    #    os.remove("dft_X_fields.h5")
    #if os.path.exists("dft_Y_empty.h5"):
    #    os.remove("dft_Y_empty.h5")
    #if os.path.exists("dft_Y_fields.h5"):
    #    os.remove("dft_Y_fields.h5")
    
    #----------------------Simulation------------------------------
    ideal=np.loadtxt('ideal_peak.txt')        # maximum source waveWIDTH
    lambda_ideal = ideal
    nfreq=1
    f_cen = 1/lambda_ideal     # source frequency center
    df = 0.05*f_cen            # source frequency width
    sy = period
    sx = dpml+sub+sep+mthick+air+dpml
    resolution = 200    #pixels/um
    diel1 = mp.Medium(index=d1)
    cell = mp.Vector3(sx,sy,0)
    #define Gaussian plane wave
    sources = [mp.Source(mp.GaussianSource(f_cen, fwidth=df),
            component=mp.Hz,
            center=mp.Vector3(-0.5*sx+dpml+0.1,0,0),
            size=mp.Vector3(0,sy,0))]
    #define pml layers
    pml_layers = [mp.Absorber(thickness=dpml, direction=mp.X)]
    supers = mp.Block(mp.Vector3(sub+sep,mp.inf,mp.inf), center=mp.Vector3(-0.5*sx+0.5*(sub+sep),0,0), material=diel1)
    geometry = [supers]
    #mp.quiet(quietval=True)
    sim = mp.Simulation(cell_size=cell,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=[mp.Mirror(mp.Y,phase=-1)],
        dimensions=2,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0))
        
    #----------------------Monitors------------------------------                     
    dfts_Y1 = sim.add_dft_fields([mp.Ey], f_cen, f_cen, 1, where=mp.Volume(center=mp.Vector3(-0.5*sx+sub+sep+mthick+0.015,0), size=mp.Vector3(0.010,sy)))
    dfts_X1 = sim.add_dft_fields([mp.Ex], f_cen, f_cen, 1, where=mp.Volume(center=mp.Vector3(-0.5*sx+sub+sep+mthick+0.015,0), size=mp.Vector3(0.010,sy)))
    #----------------------Run------------------------------
    sim.run(until_after_sources=100)
        
    #----------------------Reset------------------------------
    sim.output_dft(dfts_Y1, "dft_Y_empty")
    sim.output_dft(dfts_X1, "dft_X_empty")
    sim.reset_meep()
    grating = mp.Block(mp.Vector3(grating_t, grating_w,mp.inf), center=mp.Vector3(-0.5*sx+sub-0.5*grating_t,0,0), material=matty[int(mat)])
    metal = mp.Block(mp.Vector3(mthick,mp.inf,mp.inf), center=mp.Vector3(-0.5*sx+sub+sep+0.5*mthick,0,0), material=matty[int(mat)])
    geometry=[supers,grating,metal]
    
    
    sim = mp.Simulation(cell_size=cell,
        boundary_layers=pml_layers,
        sources=sources,
        geometry = geometry,
        symmetries=[mp.Mirror(mp.Y,phase=-1)],
        dimensions=2,
        resolution=resolution,
        k_point=mp.Vector3(0,0,0))    
        
    dfts_Y2 = sim.add_dft_fields([mp.Ey], f_cen, f_cen, 1, where=mp.Volume(center=mp.Vector3(-0.5*sx+sub+sep+mthick+0.015,0), size=mp.Vector3(0.010,sy)))
    dfts_X2 = sim.add_dft_fields([mp.Ex], f_cen, f_cen, 1, where=mp.Volume(center=mp.Vector3(-0.5*sx+sub+sep+mthick+0.015,0), size=mp.Vector3(0.010,sy)))
    sim.run(until_after_sources=400) #mp.at_beginning(mp.output_epsilon),
    sim.output_dft(dfts_Y2, "dft_Y_fields")
    sim.output_dft(dfts_X2, "dft_X_fields")
    
    #----------------------------Graph the Outputs----------------------------
    with h5py.File('dft_Y_fields.h5', 'r') as Esy:
        with h5py.File('dft_Y_empty.h5', 'r') as Eoy:
             with h5py.File('dft_X_fields.h5', 'r') as Esx:
                with h5py.File('dft_X_empty.h5', 'r') as Eox:
                    eix2 = np.array(Esx[('ex_0.i')])   
                    erx2 = np.array(Esx[('ex_0.r')])
                    eiy2 = np.array(Esy[('ey_0.i')])    
                    ery2 = np.array(Esy[('ey_0.r')])
                    E2 = (eix2**2+erx2**2)+(eiy2**2+ery2**2)
                    eix2 = np.array(Eox[('ex_0.i')])   
                    erx2 = np.array(Eox[('ex_0.r')])
                    eiy2 = np.array(Eoy[('ey_0.i')])     
                    ery2 = np.array(Eoy[('ey_0.r')]) 
                    E1 = (eix2**2+erx2**2)+(eiy2**2+ery2**2)
                    Enhance = E2/E1
                    en=np.max(np.max(Enhance))
    
    return(en)

    
pop = np.loadtxt('population.txt')
for i in range(len(pop)):
    population = pop[i,:]
    F = Simulation(population[0],population[1],population[2],population[3],population[4],population[5])
    fit = np.zeros(1)
    fit[0] = F
    np.savetxt(repr(i)+'.txt',fit)
