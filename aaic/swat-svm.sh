#!/bin/bash
#PBS --group=g-sarg
#PBS -q cq
#PBS -b 1
#PBS -l cpunum_job=1
#PBS -l elapstim_req=72:00:00
#PBS -M yoriyuki.yamagata@aist.go.jp
#PBS -m e

source activate swat-analyzer

cd ${WD}
python svm-paramsearch.py SWaT_Dataset_Normal_v0 SWaT_Dataset_Attack_v0
