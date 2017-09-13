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
    parser.add_argument('scorefile', metavar='Scores', help='Scores')
    parser.add_argument('outputfile', metavar='Outputfile', help='output file')
    args = parser.parse_args()

    logname = Path(args.logfile).stem
    logpath = (Path('output') / logname).with_suffix('.pickle')

    if logpath.exists():
        with logpath.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        sys.exit('No log file.')

    if Path(args.scorefile).exists():
        scores = pd.read_csv(args.scorefile, header=None, names=['score'])
    else:
        sys.exit()

    log = pd.concat([log_store.log, scores], axis=1, join='inner')

    n_a = log[('P6','Normal/Attack')]
    normal_dummy = pd.get_dummies(n_a)['Normal']
    log['Normal'] = normal_dummy
    log['Attack'] = 1 - normal_dummy

    log = log.sort_values(by='score', ascending=False)

    correct_detection = log['Attack'].cumsum()
    false_detection = log['Normal'].cumsum()

    precision = correct_detection / (correct_detection + false_detection)
    recall = correct_detection / (log['Attack'].sum())

    f = 2 * precision * recall / (precision + recall)

    f_max = f.idxmax()
    print('{}, {}, {}'.format(precision[f_max], recall[f_max], f[f_max]))

    scores = pd.read_csv(args.scorefile, header=None, names=['Score'])
    threshold = log['score'][f_max]
    verdict = (scores >= threshold).rename(columns={'Score':'Verdict'})
    print(verdict)
    data = pd.concat([scores, verdict], axis=1, join='inner')
    output_log = log_store.log
    output_log = pd.concat([output_log, data], axis=1, join='inner')
    output_log.to_csv(args.outputfile)
