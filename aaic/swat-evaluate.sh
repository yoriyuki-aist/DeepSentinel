#!/bin/bash
#PBS --group=g-sarg
#PBS -q cq
#PBS -b 1
#PBS -l cpunum_job=1
#PBS -l elapstim_req=8:00:00
#PBS -M yoriyuki.yamagata@aist.go.jp
#PBS -m e

#source activate swat-analyzer

cd ${WD}
python evaluate.py -n ${N_UNITS} -s 1 -a sigmoid -g -1 output/SWaT_Dataset_Normal_v0-model-1-${N_UNITS}-dropout-True-sigmoid-${ITER}-lstms.npzSWaT_Dataset_Normal_v0 SWaT_Dataset_Attack_v0 output/SWaT_Dataset_Attack_v0-scores-1-${N_UNITS}-dropout-True-sigmoid-${ITER}.csv
