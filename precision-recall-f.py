import argparse
import pickle
from pathlib import Path
import glob
import os
import sys
from storages import log_store
from storages.log_store import LogStore
import numpy as np
import pandas as pd
from sklearn import metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Precision-Recall-Fmeasure')
    parser.add_argument('logfile', metavar='Target', help='Log to be analyzed')
    parser.add_argument('normal_scorefile', metavar='NormalScores', help='Scores')
    parser.add_argument('attack_scorefile', metavar='AttackScores', help='Scores')
    parser.add_argument('output', metavar='Output', help='Output')
    args = parser.parse_args()

    print("loading log file...")
    logname = Path(args.logfile).stem
    logpath = (Path('output') / logname).with_suffix('.pickle')

    if logpath.exists():
        with logpath.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        sys.exit('No log file.')

    print("loading score files...")
    normal_scores = pd.read_csv(args.attack_scorefile, header=None, names=['score'])
    attack_scores = pd.read_csv(args.attack_scorefile, header=None, names=['score'])

    print('merge')
    log = pd.concat([log_store.log, attack_scores], axis=1, join='inner')

    print('dummy')
    n_a = log[('P6','Normal/Attack')]
    normal_dummy = pd.get_dummies(n_a)['Normal']
    log['Normal'] = normal_dummy
    log['Attack'] = 1 - normal_dummy

    print('sort')
    log = log.sort_values(by='score', ascending=False)

    correct_detection = log['Attack'].cumsum()
    false_detection = log['Normal'].cumsum()

    precision = correct_detection / (correct_detection + false_detection)
    recall = correct_detection / (log['Attack'].sum())

    f_value = 2 * precision * recall / (precision + recall)

    fp = log.apply(lambda x: normal_scores[normal_scores >= x['score']].count() / normal_scores.count(), axis=1)

    log['false_positive'] = fp
    log['precision'] = precision
    log['recall'] = recall
    log['f_value'] = f_value

    log.to_excel(args.output)
