#!/bin/bash
seq $2 $3 $4 | xargs -I "{}" python training.py --iter $5 -n "{}" -g -1 -c true  $1
