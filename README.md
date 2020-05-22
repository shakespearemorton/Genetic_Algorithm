#Genetic Algorithm for 2D MEEP layered structure

This version is designed to work on an HPC cluster, if you are running it locally, please go to the 'simulation' function and uncomment all of the checks to see if dft_fields.h5 exist. MEEP has issues overwriting these files, that are not present on the HPC cluster (who knows why).
