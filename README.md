Genetic Algorithm Optimised for 2D simulations on an HPC

Before running, edit the parameters in starter.py and run it within the folder you intend to perform the algorithm in.
1. Vari contains the names of all of your variables you're trying to optimise, and the last variable should always be fitness.
2. You can either use a range of values, or a specific dataset in your parameter space.
3. Adjust the weights to your desired output (sum to 1), and set any specific values that the system should converge to (I wanted a low reflectance at a specific wavelength).
4. Your population size should be large enough that you can get a good variability in values, but small enough that all samples in your population can be run in ~15 min. (I used a population of 49, which ran in ~6 min)
4. Your initial population will be outputted.

Edit sim.py
Given your input variables, run a simulation, and produce a fitness value. 

Adjust 'gene' in func.py
You can adjust the mutation rate, how many offspring there are, and how many random entrants there are in each iteration.

Run bashitout.sh
This will submit X generations (max 25 on the HPC) and should finish within an hour or two depending on HPC use and simulation time.
