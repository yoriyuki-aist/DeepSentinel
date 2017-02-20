#!/bin/bash
seq $1 $2 $3 | xargs -I "{}" python rnn-anomaly-detector.py --iter $4 -n "{}" -g 0 -c true ../../data/aquarium/log.csv $5
