#!/bin/bash
#PBS --group=g-sarg
#PBS -q gq
#PBS -b 1
#PBS -l cpunum_job=40
#PBS -l gpunum_job=8
#PBS -l elapstim_req=72:00:00
#PBS -M yoriyuki.yamagata@aist.go.jp
#PBS -m e

module unload tensorflow/0.12/cpu_python2.7
source activate swat-analyzer
# echo $CUDA_VISIBLE_DEVICES
# arrGPU=(${CUDA_VISIBLE_DEVICES//,/ })
# GPU=${arrGPU[0]}
# echo $GPU

cd /home/yoriyuki/Factory/swat-analyzer
python training.py -n 200 -i 100 -s 1 -c true -d true -a relu -g 0 SWaT_Dataset_Normal_v0 &
python training.py -n 300 -i 100 -s 1 -c true -d true -a relu -g 1 SWaT_Dataset_Normal_v0 &
python training.py -n 400 -i 100 -s 1 -c true -d true -a relu -g 2 SWaT_Dataset_Normal_v0 &
python training.py -n 500 -i 100 -s 1 -c true -d true -a relu -g 3 SWaT_Dataset_Normal_v0 &
python training.py -n 200 -i 100 -s 1 -c true -d true -a sigmoid -g 4 SWaT_Dataset_Normal_v0 &
python training.py -n 300 -i 100 -s 1 -c true -d true -a sigmoid -g 5 SWaT_Dataset_Normal_v0 &
python training.py -n 400 -i 100 -s 1 -c true -d true -a sigmoid -g 6 SWaT_Dataset_Normal_v0 &
python training.py -n 500 -i 100 -s 1 -c true -d true -a sigmoid -g 7 SWaT_Dataset_Normal_v0 &

wait
