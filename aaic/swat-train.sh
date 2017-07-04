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
module load chainer/1.24.0/gpu_python3.4
echo $CUDA_VISIBLE_DEVICES
arrGPU=(${CUDA_VISIBLE_DEVICES//,/ })
GPU=${arrGPU[0]}
echo $GPU

cd /home/yoriyuki/Factory/swat-analyzer
python training.py -n ${N_UNITS} -i ${ITER} -s 1 -c true -d true -a ${ACTIVATION} -g $GPU SWaT_Dataset_Normal_v0

wait
