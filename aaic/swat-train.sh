#!/bin/bash
#PBS --group=g-sarg
#PBS -q gq
#PBS -b 1
#PBS -l cpunum_job=5
#PBS -l gpunum_job=1
#PBS -l elapstim_req=72:00:00
#PBS -l memsz_job=30gb
#PBS -M yoriyuki.yamagata@aist.go.jp
#PBS -m e

source activate swat-analyzer

arrGPU=(${CUDA_VISIBLE_DEVICES//,/ })
GPU=${arrGPU[0]}

python training.py -n 100 -i 100 -s 1 -c true -d true -a sigmoid -g $GPU SWaT_Dataset_Normal_v0 &
python training.py -n 200 -i 100 -s 1 -c true -d true -a sigmoid -g $GPU SWaT_Dataset_Normal_v0 &
python training.py -n 300 -i 100 -s 1 -c true -d true -a sigmoid -g $GPU SWaT_Dataset_Normal_v0 &
python training.py -n 400 -i 100 -s 1 -c true -d true -a sigmoid -g $GPU SWaT_Dataset_Normal_v0 &
python training.py -n 500 -i 100 -s 1 -c true -d true -a sigmoid -g $GPU SWaT_Dataset_Normal_v0 &

wait
