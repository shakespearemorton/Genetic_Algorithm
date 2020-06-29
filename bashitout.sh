#!/bin/bash

go=$(qsub run.pbs)
echo $go
    
for gen in {0..40..1}
do

    go=$(qsub -W depend=afterany:$go run.pbs)
    echo $go
done

qsub -W depend=afterok:$go run_fitter.pbs
