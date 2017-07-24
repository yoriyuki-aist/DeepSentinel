import argparse
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from pathlib import Path
from storages.log_store import LogStore
from tqdm import tqdm
from functools import reduce
import time
# from log_store import LogStore

if __name__ == '__main__':
    start_time = time.time ()
    default_window = 1
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('model', metavar='model',
                        help='Pickled model file (without output/ or extension)')
    parser.add_argument('training', metavar='training',
                        help='Data set used for training (needed for normalization)')
    parser.add_argument('target',
                        help='Target log file')
    parser.add_argument('--log', '-l', default=sys.stdout,
                        help='Log file (for stats like F-score)')
    parser.add_argument('--window', '-w', type=int, default=default_window,
                        help='How many entries to mash into one sample [default: {}]'.format (default_window))
    parser.add_argument('--debug', metavar='N', type=int, default=0,
                        help='Level of debugging output')

    args = parser.parse_args()

    if not Path('output').exists():
        os.mkdir('output')
    else:
        if not Path('output').is_dir():
            sys.exit('output is not a directory.')

    print ('loading model file...')
    modelfile = Path(args.model)
    savefile = (modelfile.parent / (modelfile.stem + '-results.pickle'))

    with modelfile.open ('rb') as f:
        model = pickle.load (f)

    normallogname = Path(args.training).stem
    normallogstore = (Path('output') / normallogname).with_suffix('.pickle')

    print("loading normalization data from training set...")
    print (normallogstore)
    if normallogstore.exists():
        with normallogstore.open(mode='rb') as normallogstorefile:
            normal_log_store = pickle.load(normallogstorefile)
    else:
        normal_log_store = LogStore(args.training)
        with normallogstore.open(mode='wb') as f:
            pickle.dump(normal_log_store, f)

    print("loading log file...")

    targetlogname = Path (args.target).stem
    targetlogstore = (Path ('output') / targetlogname).with_suffix ('.pickle')
    print ('target =', targetlogstore)
    if targetlogstore.exists():
        with targetlogstore.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        log_store = LogStore(targetlogstore)

    print("preprocessing...")

    pp_start_time = time.time ()

    labels = np.concatenate (log_store.train_l_seqs)
    values = np.concatenate (log_store.train_v_seqs)
    positions = np.concatenate (log_store.train_p_seqs)
    data = np.concatenate ([values, positions], axis=1)

    is_normal = (labels == 'Normal')

    if args.debug >= 1:
        print ('data :', data.shape)
    if args.debug >= 2:
        print (data)
    window = [data[i:-(args.window-i)] for i in range (args.window)]
    is_normal_window = [is_normal[i:-(args.window-i)] for i in range (args.window)]
    data = np.concatenate (window, axis=1)
    is_normal = reduce (np.logical_and, is_normal_window)

    if args.debug >= 4:
        print ('window\n', window)
        print ('is_normal_window\n', is_normal_window)
        print ('data\n', data)
        print ('is_normal\n', is_normal)

    print("start evaluating...")

    pp_end_time = time.time ()

    if args.debug >= 1:
        print ('data :', data.shape)
    if args.debug >= 4:
        print ('data\n', data)

    eval_start_time = time.time ()

    if savefile.exists ():
        print ('loading existing evaluation results...')
        with savefile.open ('rb') as f:
            (is_normal, pred) = pickle.load (f)
    else:
        pred = model.predict (data)

        with savefile.open ('wb') as f:
            pickle.dump ((is_normal, pred), f)

    eval_end_time = time.time ()

    print ('compiling results...')

    stat_start_time = time.time ()

    # NB: prediction values are -1 if attack, +1 if normal
    pred_normal = (pred == 1)
    pred_attack = 1 - pred_normal
    is_attack = 1 - is_normal
    n_false_pos = np.count_nonzero (pred_attack * is_normal)
    n_true_pos  = np.count_nonzero (pred_attack * is_attack)
    n_false_neg = np.count_nonzero (pred_normal * is_attack)
    n_true_neg  = np.count_nonzero (pred_normal * is_normal)
    n_pred_normal = np.count_nonzero (pred_normal)
    n_pred_attack = pred_normal.size - n_pred_normal
    n_normal = np.count_nonzero (is_normal)
    n_attack = is_normal.size - n_normal
    n_total = is_normal.size

    n_correct = n_true_pos + n_true_neg
    #n_incorrect = n_false_pos + n_false_neg

    n_pred_pos = n_true_pos + n_false_pos
    if n_pred_pos == 0:
        precision = float ('nan')
    else:
        precision = n_true_pos / (n_true_pos + n_false_pos)
    if n_attack == 0:
        recall = float ('nan')
    else:
        recall = n_true_pos / n_attack
    if precision + recall == 0:
        f_score = float ('nan')
    else:
        f_score = 2 * precision * recall / (precision + recall)

    stat_end_time = time.time ()
    end_time = stat_end_time

    if not (hasattr (args.log, 'write')):
        args.log = open (args.log, 'w')
    def output (x):
        print (x, file=args.log)

    output ('model:    {}'.format (modelfile))
    output ('training: {}'.format (normallogstore))
    output ('target:   {}'.format (args.target))
    output ('results saved in: {}'.format (savefile))
    output ('')
    output ('preprocessing time:    {}'.format (pp_end_time - pp_start_time))
    output ('evaluation time:       {}'.format (eval_end_time - eval_start_time))
    output ('stat compilation time: {}'.format (stat_end_time - stat_start_time))
    output ('total time:            {}'.format (end_time - start_time))
    output ('')
    output ('normal entries:      {} / {} = {}'.format (
        n_normal, n_total, n_normal / n_total))
    output ('attack entries:      {} / {} = {}'.format (
        n_attack, n_total, n_attack / n_total))
    output ('predicted normal:    {} / {} = {}'.format (
        n_pred_normal, n_total, n_pred_attack / n_total))
    output ('predicted attack:    {} / {} = {}'.format (
        n_pred_attack, n_total, n_pred_attack / n_total))
    output ('correct predictions: {} / {} = {}'.format (
        n_correct, n_total, n_correct / n_total))
    output ('true positives:  {}'.format (n_true_pos))
    output ('true negatives:  {}'.format (n_true_neg))
    output ('false positives: {}'.format (n_false_pos))
    output ('false negatives: {}'.format (n_false_neg))
    output ('precision: {}'.format (precision))
    output ('recall:    {}'.format (recall))
    output ('F-score:   {}'.format (f_score))
