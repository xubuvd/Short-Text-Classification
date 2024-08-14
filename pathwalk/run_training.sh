#!/bin/bash

ulimit -c unlimited
ulimit -n 8192

#dataset=(MPQA  CVQD  DanMu  MPQA  MR  SST1  SST2  SUBJ  TREC)
dataset=(DanMu)

for data in ${dataset[@]}
do
    echo $data

    CUDA_VISIBLE_DEVICES=1 python train.py \
        --gpu-ids 1 --validate \
        --config-yml configs/DanMu.yml \
        --save-dirpath ./output/2022061405
done

