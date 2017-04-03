#!/bin/bash
TARGET=$1
START=$2
STEP=$3
FINISH=$4
shift 4
seq $START $STEP $FINISH | xargs -I "{}" python training.py -n "{}" $* $TARGET
