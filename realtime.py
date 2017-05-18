import argparse
import pickle
from pathlib import Path
import glob
import os
import sys
import itertools
from storages import log_store
from storages.log_store import LogStore
import log_model
from log_model import logModel
from log_model.logModel import LogModel
import numpy as np
import pandas as pd
from sklearn import metrics

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

value_labels = [('P1', 'FIT101'),
('P1', 'LIT101'),
('P2', 'AIT201'),
('P2', 'AIT202'),
('P2', 'AIT203'),
('P2', 'FIT201'),
('P3', 'DPIT301'),
('P3', 'FIT301'),
('P3', 'LIT301'),
('P4', 'AIT401'),
('P4', 'AIT402'),
('P4', 'FIT401'),
('P4', 'LIT401'),
('P5', 'AIT501'),
('P5', 'AIT502'),
('P5', 'AIT503'),
('P5', 'AIT504'),
('P5', 'FIT501'),
('P5', 'FIT502'),
('P5', 'FIT503'),
('P5', 'FIT504'),
('P5', 'PIT501'),
('P5', 'PIT502'),
('P5', 'PIT503'),
('P6', 'FIT601') ]

discrete_labels = [
('P1', 'MV101'),
('P1', 'P101'),
('P1', 'P102'),
('P2', 'MV201'),
('P2', 'P201'),
('P2', 'P202'),
('P2', 'P203'),
('P2', 'P204'),
('P2', 'P205'),
('P2', 'P206'),
('P3', 'MV301'),
('P3', 'MV302'),
('P3', 'MV303'),
('P3', 'MV304'),
('P3', 'P301'),
('P3', 'P302'),
('P4', 'P401'),
('P4', 'P402'),
('P4', 'P403'),
('P4', 'P404'),
('P4', 'UV401'),
('P5', 'P501'),
('P5', 'P502'),
('P6', 'P601'),
('P6', 'P602'),
('P6', 'P603'),]

label_order = [
('P1', 'FIT101'),
('P1', 'LIT101'),
('P1', 'MV101'),
('P1', 'P101'),
('P1', 'P102'),
('P2', 'AIT201'),
('P2', 'AIT202'),
('P2', 'AIT203'),
('P2', 'FIT201'),
('P2', 'MV201'),
('P2', 'P201'),
('P2', 'P202'),
('P2', 'P203'),
('P2', 'P204'),
('P2', 'P205'),
('P2', 'P206'),
('P3', 'DPIT301'),
('P3', 'FIT301'),
('P3', 'LIT301'),
('P3', 'MV301'),
('P3', 'MV302'),
('P3', 'MV303'),
('P3', 'MV304'),
('P3', 'P301'),
('P3', 'P302'),
('P4','AIT401'),
('P4','AIT402'),
('P4','FIT401'),
('P4','LIT401'),
('P4','P401'),
('P4','P402'),
('P4','P403'),
('P4','P404'),
('P4','UV401'),
('P5', 'AIT501'),
('P5', 'AIT502'),
('P5', 'AIT503'),
('P5', 'AIT504'),
('P5', 'FIT501'),
('P5', 'FIT502'),
('P5', 'FIT503'),
('P5', 'FIT504'),
('P5', 'P501'),
('P5', 'P502'),
('P5', 'PIT501'),
('P5', 'PIT502'),
('P5', 'PIT503'),
('P6', 'FIT601'),
('P6', 'P601'),
('P6', 'P602'),
('P6', 'P603'),
]

def batch_seq(seq, chunk_num):
    seq = seq[0:len(seq) // chunk_num * chunk_num]
    chunked = np.split(seq, chunk_num)
    return np.stack(chunked, axis=-1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute outlier factors and evaluate them')
    parser.add_argument('normal', metavar='NormalLog', help='Normal log')
    parser.add_argument('model', metavar='Modelile', help='Model file')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_units', '-n', type=int, default=0,
                        help='Number of n_units')
    parser.add_argument('--lstm', '-s', type=int, default=0,
                    help='the height of stacked LSTMs')
    args = parser.parse_args()

    eprint('Creating output directory...')
    if not Path('output').exists():
        os.mkdir('output')
    else:
        if not Path('output').is_dir():
            sys.exit('output is not a directory.')

    normallogname = Path(args.normal).stem
    normallogstore = (Path('output') / normallogname).with_suffix('.pickle')

    eprint("loading normal log file...")
    if normallogstore.exists():
        with normallogstore.open(mode='rb') as normallogstorefile:
            normal_log_store = pickle.load(normallogstorefile)
    else:
        normal_log_store = LogStore(args.normal)
        with normallogstore.open(mode='wb') as f:
            pickle.dump(normal_log_store, f)

    normal = normal_log_store
    zscore = lambda x: (x - normal.log[x.name].mean()) / normal.log[x.name].std()

    eprint('loading model file...')
    if not Path(args.model).exists():
        sys.exit('model does not exists.')
    else:
        log_model = LogModel(normal_log_store, lstm_num=args.lstm, n_units=args.n_units, gpu=args.gpu, directory='output/', logLSTM_file=args.model)

    eprint("start evaluating...")
    log = pd.read_csv(sys.stdin, chunksize=1, header=None, names=label_order, delimiter=',', index_col=False)
    log = itertools.islice(log, 1, None)
    seq = ((batch_seq(row[discrete_labels].values, 1)[0], batch_seq(row[value_labels].apply(zscore).values, 1)[0]) for row in log)
    cur, nt = itertools.tee(seq)
    nt = itertools.islice(nt, 1, None)
    data = zip(cur, nt)

    log_model.model.reset_state()
    scores = (log_model.model.eval(cur, nt).data for cur, nt in data)

    for score in scores:
        print(score)
