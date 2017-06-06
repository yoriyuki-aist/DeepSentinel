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
    scores = list(log_model.eval(log_store))
    scores = np.array(scores, np.float32)
    log = log_store.log
    log['score'] = pd.Series(scores, log.index[:-1]).shift(1)

    log['score'].to_csv(args.output)
