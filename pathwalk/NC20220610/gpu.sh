#!/bin/bash

nvidia-smi --debug

#nvidia-smi -l 2
watch --color -n1 gpustat -cpu --debug

#执行fuser -v /dev/nvidia* 发现僵尸进程（连号的）
#fuser -v /dev/nvidia*

