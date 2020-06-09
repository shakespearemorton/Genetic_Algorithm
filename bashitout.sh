#!/bin/bash

go=$(qsub run.pbs)
echo $go
    
for gen in {0..20..1}
do
    fit=$(qsub -W depend=afterany:$go run_fitter.pbs)
    echo $fit
    go=$(qsub -W depend=afterany:$fit run.pbs)
    echo $go
done

qsub -W depend=afterok:$go run_fitter.pbs
