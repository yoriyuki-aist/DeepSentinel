#!/bin/bash
#PBS --group=g-sarg
#PBS -q gq
#PBS -b 1
#PBS -l cpunum_job=1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=72:00:00
#PBS -M yoriyuki.yamagata@aist.go.jp
#PBS -m e

module switch cuda/8.0 cuda/8.0.61+cudnn-6.0.21+nccl-1.3.4-1
source activate swat-analyzer

cd ${WD}
python training.py -n ${N_UNITS} -i ${ITER} -s ${LSTM} -c true -d true -a ${ACTIVATION} -g 0 SWaT_Dataset_Normal_v0
