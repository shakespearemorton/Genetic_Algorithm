#Genetic Algorithm for 2D MEEP layered structure

This version is designed to work on an HPC cluster, if you are running it locally, please go to the 'simulation' function and uncomment all of the checks to see if dft_fields.h5 exist. MEEP has issues overwriting these files, that are not present on the HPC cluster (who knows why).

Specifically, this code is set up to optimise the separation distance between two metals for a fabry perot mirror, and the thickness of the top metal for LRSPP coupling. The gap in the middle is present so that an incident EM wave will excite LSPR and SPP on the corners of the structure. 

Outputs include: Structure (in the form of dielectric constants), E-field, Extinction Spectrum, and how it has evolved through the GA
