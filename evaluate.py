import argparse
import pickle
from pathlib import Path
import glob
import os
import sys
from storages import log_store
from storages.log_store import LogStore
import log_model
from log_model import logModel
from log_model.logModel import LogModel
import numpy as np
import pandas as pd
from sklearn import metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute outlier factors and evaluate them')
    parser.add_argument('normal', metavar='NormalLog', help='Normal log')
    parser.add_argument('model', metavar='Modelile', help='Model file')
    parser.add_argument('logfile', metavar='Target', help='Log to be analyzed')
    parser.add_argument('output', metavar='Output', help='Output')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_units', '-n', type=int, default=0,
                        help='Number of n_units')
    parser.add_argument('--lstm', '-s', type=int, default=0,
                    help='the height of stacked LSTMs')
    args = parser.parse_args()

    print('Creating output directory...')
    if not Path('output').exists():
        os.mkdir('output')
    else:
        if not Path('output').is_dir():
            sys.exit('output is not a directory.')

    normallogname = Path(args.normal).stem
    normallogstore = (Path('output') / normallogname).with_suffix('.pickle')

    print("loading normal log file...")
    if normallogstore.exists():
        with normallogstore.open(mode='rb') as normallogstorefile:
            normal_log_store = pickle.load(normallogstorefile)
    else:
        normal_log_store = LogStore(args.normal)
        with normallogstore.open(mode='wb') as f:
            pickle.dump(normal_log_store, f)

    logname = Path(args.logfile).stem
    logstore = (Path('output') / logname).with_suffix('.pickle')

    print("loading log file...")
    if logstore.exists():
        with logstore.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        log_store = LogStore(args.logfile, normal_log_store)
        with logstore.open(mode='wb') as f:
            pickle.dump(log_store, f)

    print('loading model file...')
    if not Path(args.model).exists():
        sys.exit('model does not exists.')
    else:
        log_model = LogModel(log_store, lstm_num=args.lstm, n_units=args.n_units, gpu=args.gpu, directory='output/', logLSTM_file=args.model)

    print("start evaluating...")
    scores = list(log_model.eval(log_store.positions_seq, log_store.values_seq))
    scores = np.array(scores, np.float32)
    log = log_store.log
    log['score'] = pd.Series(scores, log.index[:-1]).shift(1)

    n_a = log[('P6','Normal/Attack')]
    normal_dummy = pd.get_dummies(n_a)['Normal']
    log['Normal'] = normal_dummy
    log['Attack'] = 1 - normal_dummy

    print(log)
    log = log.sort_values(by='score', ascending=False)

    correct_detection = log['Attack'].cumsum()
    false_detection = log['Normal'].cumsum()

    precision = correct_detection / (correct_detection + false_detection)
    recall = correct_detection / (log['Attack'].sum())
    fp = false_detection / (log['Normal'].sum())

    f_value = 2 * precision * recall / (precision + recall)

    log['precision'] = precision
    log['recall'] = recall
    log['false_positive'] = fp
    log['f_value'] = f_value

    max_f_index = f_value.idxmax()
    print(max_f_index)
    k = log['score'].loc[max_f_index]
    f = log['f_value'].loc[max_f_index]
    p = log['precision'].loc[max_f_index]
    r = log['recall'].loc[max_f_index]
    print("Threshold: {}, Precision: {}, Recall: {}, F value: {}".format(k, p, r, f))

    judgements = pd.cut(log['score'].values, [log['score'].min(), k, log['score'].max()+1], right=False, labels=['N', 'A'])
    log['judge'] = judgements

    print('AUC: {}'.format(metrics.auc(fp.values, recall.values)))
    log.to_excel(args.output)
