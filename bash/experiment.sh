	#!/bin/bash
# 0.01 0.05 0.07 0.09 0.1 0.11 0.13 0.15 0.2
# upsilon = [1 1.2 1.4 1.6 1.8 2.0]
# gamma = [0.01 0.05 0.07 0.09 0.1 0.11 0.13 0.15 0.2]
# noise = [0.05 0.1 0.15 0.2 0.25 0.30 0.35 0.40]
cd ../scripts
for lam in 0.05 0.07 0.09 0.1 0.11 0.15 0.2
do
    for upsi in 1 1.2 1.4 1.6 1.8 2.0
    do
        for gam in 0.05 0.07 0.09 0.1 0.11 0.15 0.2
        do
            for noise in 0.1 0.15 0.2 0.25 0.30
            do
                python gen_config.py ../config/envs/point_mass2d.yaml elipse -l $lam -g $gam -u $upsi -n $noise
                python main.py /tmp/config.yaml /tmp/task.yaml -s 300 -l ## -g ../gifs/exp_l$lam_g$gam_u$upsi_n$noise.gif
            done
        done
    done
done
