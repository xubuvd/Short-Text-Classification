#!/bin/bash

ulimit -c unlimited
ulimit -n 8192
#MPQA  CVQD  DanMu  MPQA  MR  SST1  SST2  SUBJ  TREC
#DPCNN FastText TextCNN TextLSTM TextRCNN TextRNN_Att TextRNN Transformer

methods=(DPCNN FastText TextCNN TextLSTM TextRCNN TextRNN_Att TextRNN Transformer)

#methods=(TextRNN)

for element in ${methods[@]}
do
#for((i=1;i<=10;i++));
#do
echo ${element}_${i}
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "CR" --pad 10 --word > log_${element}_2022060901_CR.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "MPQA" --pad 10 --word > log_${element}_2022060901_MPQA.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "SST1" --pad 10 --word > log_${element}_2022060901_SST1.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "SST2" --pad 10 --word > log_${element}_2022060901_SST2.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "SUBJ" --pad 10 --word > log_${element}_2022060901_SUBJ.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "TREC" --pad 10 --word > log_${element}_2022060901_TREC.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "MR" --pad 10 --word > log_${element}_2022060901_MR.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "DanMu" --pad 10  > log_${element}_2022060901_DanMu.log 2>&1
CUDA_VISIBLE_DEVICES=2 python run.py --model $element --dataset "CVQD" --pad 10  > log_${element}_2022060901_CVQD.log 2>&1
#done

done

#--overfit

