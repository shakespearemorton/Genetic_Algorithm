import meep as mp
import numpy as np
from meep.materials import Au,Al,Ag
import random
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import seaborn as sb
import os

def simulation(f_cen,df,fmin,fmax,sy,dpml,air,sx,resolution,nfreq,geometry,init_refl_data,init_tran_flux,n,grate,THICKNESS):
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
    mp.quiet(quietval=True)
    if n ==0:
        if os.path.exists('dft_Z_empty.h5'):
            os.remove('dft_Z_empty.h5')
        sim = mp.Simulation(cell_size=cell,
                boundary_layers=pml_layers,
                sources=sources,
                dimensions=2,
                resolution=resolution,
                k_point=mp.Vector3())
        
        #----------------------Monitors------------------------------
        refl = sim.add_flux(f_cen, df, nfreq, refl_fr)
        tran = sim.add_flux(f_cen,df, nfreq, tran_fr)                        
        dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,-sy*0.5+THICKNESS[4]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+THICKNESS[3]+grate,0), size=mp.Vector3(sx,grate*2)))
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
                dimensions=2,
                resolution=resolution,
                k_point=mp.Vector3())
        
        refl = sim.add_flux(f_cen, df, nfreq, refl_fr)
        tran = sim.add_flux(f_cen,df, nfreq, tran_fr)
        sim.load_minus_flux_data(refl,init_refl_data)
        dfts_Z = sim.add_dft_fields([mp.Ez], fmin, fmax, nfreq, where=mp.Volume(center=mp.Vector3(0,-sy*0.5+THICKNESS[4]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+THICKNESS[3]+grate,0), size=mp.Vector3(sx,grate*2)))
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
        en = enhance(Rs)
        get=en,np.min(Rs)
        #plt.figure()
        #plt.plot(wl,Rs,'bo-',label='reflectance')
        #eps = h5py.File('eps-000000.00.h5', 'r')
        #eps = eps.get('eps').value
        #Enhance = np.rot90(eps)
        #heat_map = sb.heatmap(Enhance,cmap='plasma',xticklabels=False, yticklabels=False)
        
            
    return(get)

def geom(THICKNESS,sx,sy,grate,g,d1,d2,material,col,row,matter,dpml):
    Glass = mp.Medium(epsilon=g)
    diel1 = mp.Medium(epsilon=d1)
    diel2 = mp.Medium(epsilon=d2)
    mat = [Au,Al,Ag,Glass,diel1,diel2]
    #----------------------------Geometry------------------------------
    substrate = mp.Block(mp.Vector3(mp.inf, THICKNESS[0],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[0]*0.5+dpml,0),  material=mat[material[0]])
    mirror = mp.Block(mp.Vector3(mp.inf, THICKNESS[1],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[1]*0.5+THICKNESS[0]+dpml,0),  material=mat[material[1]])
    cytop = mp.Block(mp.Vector3(mp.inf,THICKNESS[2],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[2]*0.5+THICKNESS[0]+THICKNESS[1]+dpml,0),  material=mat[material[2]])
    gold = mp.Block(mp.Vector3(mp.inf,THICKNESS[3],mp.inf), center=mp.Vector3(0, -sy*0.5+THICKNESS[3]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+dpml,0),  material=mat[material[3]])
    water = mp.Block(mp.Vector3(mp.inf, THICKNESS[4],mp.inf), center=mp.Vector3(0,-sy*0.5+THICKNESS[4]*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+THICKNESS[3]+dpml,0),  material=mat[material[4]])
    geometry = [substrate,mirror,cytop,gold,water]
    x=grate
    
    if row > 1:
        k=0
        for j in range(row):
            posy = -sy*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+THICKNESS[3]+((j)*grate)+(grate*0.5)+dpml
            x=grate
            for i in range(col):
                posx = (-0.5*sx)+(x-grate*0.5)
                z=int(matter[k])
                geometry.append(mp.Block(mp.Vector3(grate, grate,mp.inf), center=mp.Vector3(posx,posy,0), material=mat[z]))
                x+=grate
                k+=1
    else:
        posy = -sy*0.5+THICKNESS[0]+THICKNESS[1]+THICKNESS[2]+THICKNESS[3]+(0.5*grate)+dpml
        for i in range(col):
            posx = (-0.5*sx)+(x-grate*0.5)
            z=int(matter[i])
            geometry.append(mp.Block(mp.Vector3(grate, grate,mp.inf), center=mp.Vector3(posx,posy,0), material=mat[z]))
            x+=grate
    return(geometry)

def roulette(R,pop,sizer):
    F = R
    F = (F-np.min(F))/(np.max(F)-np.min(F))
    idx = np.argsort(F)  
    F = np.array(F)[idx]
    pop = np.array(pop)[idx]
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
    
def genes(pop,population,sizer,mat_options):
    if population < 2:
        pass
    else:
        parents = pop
        pop = np.zeros((population,sizer))
        pop[0,:]=np.random.choice(mat_options, size=(sizer))
        for i in range(sizer):
            doh = random.uniform(0,1)
            if doh < 0.1:
                pop[1,i] = np.random.choice(mat_options,1)
            else:
                pop[1,i] = parents[2,i]
        for i in range(sizer):
            doh = random.uniform(0,1)
            if doh < 0.2:
                pop[2,i] = np.random.choice(mat_options,1)
            else:
                pop[2,i] = parents[2,i]
        for i in range(sizer):
            pop[3,i] = parents[2,i]
        if population > 4:
            for j in range(population-4):
               off = randint(0,sizer)
               x=j+4
               for i in range(sizer):
                   if i <= off:
                       pop[x,i]=parents[1,i]
                      
                   else:
                       pop[x,i]=parents[0,i]
    return(pop)

def posty(saint,best_grating,progress,sizer):
    pic = np.zeros([200,100*sizer],dtype=np.uint8)
    on=1
    for j in range(2):
        x=0
        y=99
        for i in range(sizer):
            if on ==1:
                pic[0:99,x:y] = saint[i]* 255 
                x+=100
                y+=100
            else:
                pic[100:200,x:y] = 255 * best_grating[i]
                x+=100
                y+=100
        on=0
    print(pic)
    img = Image.fromarray(pic)
    img.save('LOOK.png')      
    return()

def enhance(R):
    Es = h5py.File('dft_Z_fields.h5', 'r')
    Eo = h5py.File('dft_Z_empty.h5', 'r')
    
    ei2 = Es.get('ez_'+repr(np.argmax(R))+'.i').value     
    er2 = Es.get('ez_'+repr(np.argmax(R))+'.r').value 
    E2 = ei2**2+er2**2
    ei1 = Eo.get('ez_'+repr(np.argmax(R))+'.i').value     
    er1 = Eo.get('ez_'+repr(np.argmax(R))+'.r').value 
    E1 = ei1**2+er1**2
    Enhance = E2/E1
    return(np.max(np.max(Enhance)))    

def imagine(saint,par,sizer):
    for i in range(sizer):
        saint[i]=saint[i]+par[2,i]
    return(saint)