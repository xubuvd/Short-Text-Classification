#!/bin/bash

ulimit -c unlimited
ulimit -n 8192
for (( c=1; c<=1; c++ ))
do
    ##CR  CVQD  DanMu  MPQA  MR  SST1  SST2  SUBJ  THUCNews  TREC
    echo "Welcome $c times"
    CUDA_VISIBLE_DEVICES=1 python run.py \
        --dataset R8 \
        --use_LDA \
        --batchSize 64 \
        --lr 0.001 \
        --dropout 0.3 \
        --l2 1e-6 \
        --epoch 20

    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

#--overfit

