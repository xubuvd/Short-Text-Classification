#!/bin/bash

ulimit -c unlimited
ulimit -n 8192
for (( c=1; c<=1; c++ ))
do
    echo "Welcome $c times"
    CUDA_VISIBLE_DEVICES=0 python evaluate.py \
        --gpu-ids 0 \
        --validate \
        --config-yml configs/pathwalk.yml \
        --load_pthpath  ./output/2022040107/checkpoint_19.pth

    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

#--overfit

