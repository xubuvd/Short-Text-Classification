#!/bin/bash

ulimit -c unlimited
ulimit -n 8192
for (( c=1; c<=1; c++ ))
do
    #CR  CVQD  DanMu  MPQA  MR  SST1  SST2  SUBJ  THUCNews  TREC
    echo "Welcome $c times"
    nohup python generate_lda.py --topics 2 --dataset CR > CR_lda.log 2>&1 &
    nohup python generate_lda.py --topics 26 --dataset CVQD > CVQD_lda.log 2>&1 &
    nohup python generate_lda.py --topics 2 --dataset DanMu > DanMu_lda.log 2>&1 &
    nohup python generate_lda.py --topics 2 --dataset MPQA > MPQA_lda.log 2>&1 &
    nohup python generate_lda.py --topics 2 --dataset MR > MR_lda.log 2>&1 &
    nohup python generate_lda.py --topics 5 --dataset SST1 > SST1_lda.log 2>&1 &
    nohup python generate_lda.py --topics 2 --dataset SST2 >SST2_lda.log 2>&1 &
    nohup python generate_lda.py --topics 2 --dataset SUBJ >SUBJ_lda.log 2>&1 &
    nohup python generate_lda.py --topics 2 --dataset TREC > TREC_lda.log 2>&1 &

    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

#--overfit

