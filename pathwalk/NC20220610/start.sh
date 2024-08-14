#!/bin/bash

ulimit -c unlimited
ulimit -n 8192

#dataset=(MPQA  CVQD  DanMu  MPQA  MR  SST1  SST2  SUBJ  TREC)
dataset=(70)
#dataset=(DanMu)

for data in ${dataset[@]}
do
#for((i=1;i<=10;i++));
#do

echo $data
#CUDA_VISIBLE_DEVICES=0 python train.py --gpu-ids 0 --validate --select_indegree_num $data --config-yml configs/DanMu.yml --save-dirpath  ./output/2022052705_${data} > log_PathWalk_2022052705_58_gpu0_DanMu_300d5_select_indegree_num_${data}.log 2>&1

CUDA_VISIBLE_DEVICES=1 python train.py --gpu-ids 1 --validate --config-yml configs/DanMu.yml --save-dirpath ./output/2022060902 > log_PathWalk_2022060902_DanMu_30pad.log 2>&1 &

#done
done

#--overfit

