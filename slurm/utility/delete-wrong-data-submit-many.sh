#!/bin/bash

# Submit deletion for inputs
sbatch -J "input" /home/users/mendrika/Zambia-Intercomparison/slurm/utility/delete-wrong-data.sh \
    /gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa/inputs_t0

# Submit deletion for all target lead times
for i in 0 1 3 6; do 
    sbatch -J "targets_t$i" /home/users/mendrika/Zambia-Intercomparison/slurm/utility/delete-wrong-data.sh \
        "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa/targets_t$i"
done
