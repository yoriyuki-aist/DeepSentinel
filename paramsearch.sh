#!/bin/bash
seq $2 $3 $4 | xargs -I "{}" python training.py --iter $5 -n "{}" --lstm 1 -g 0 -c true  $1
