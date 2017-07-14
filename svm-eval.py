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
# from log_store import LogStore

if __name__ == '__main__':
    default_gap = 1
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('model', metavar='model',
                        help='Pickled model file (without output/ or extension)')
    parser.add_argument('training', metavar='training',
                        help='Data set used for training (needed for normalization)')
    parser.add_argument('target',
                        help='Target log file')
    parser.add_argument('--log', '-l', default=sys.stdout,
                        help='Log file (for stats like F-score)')
    parser.add_argument('--gap', '-G', type=int, default=default_gap,
                        help='How many entries forward to predict [default: {}]'.format (default_gap))
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
    if targetlogstore.exists():
        with targetlogstore.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        log_store = LogStore(targetlogstore)

    print("preprocessing...")

    labels = np.concatenate (log_store.train_l_seqs)
    values = np.concatenate (log_store.train_v_seqs)
    positions = np.concatenate (log_store.train_p_seqs)
    data = np.concatenate ([values, positions], axis=1)

    if args.debug >= 1:
        print ('data :', data.shape)
    if args.debug >= 2:
        print (data)
    cur = data[:-args.gap]
    nxt = data[args.gap:]
    data = np.concatenate ([cur, nxt], axis=1)
    labels = labels[args.gap:]

    if args.debug >= 4:
        print ('cur\n', cur)
        print ('nxt\n', nxt)
        print ('data\n', data)

    print("start evaluating...")

    if args.debug >= 1:
        print ('data :', data.shape)
    if args.debug >= 4:
        print ('data\n', data)

    pred = model.predict (data)

    print ('compiling results...')

    savefile = (modelfile.parent / (modelfile.stem + '-results.pickle'))
    with savefile.open ('wb') as f:
        pickle.dump (pred, f)

    is_normal = (labels == 'Normal')
    # NB: positive = attack, negative = normal
    pred_normal = (pred == -1)
    n_false_pos = np.count_nonzero ((1 - pred_normal) * is_normal)
    n_true_pos  = np.count_nonzero ((1 - pred_normal) * (1 - is_normal))
    n_false_neg = np.count_nonzero (pred_normal * (1 - is_normal))
    n_true_neg  = np.count_nonzero (pred_normal * is_normal)
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
    f_score = 2 * precision * recall / (precision + recall)

    if not (hasattr (args.log, 'write')):
        args.log = open (args.log, 'w')
    def output (x):
        print (x, file=args.log)

    output ('model:    {}'.format (modelfile))
    output ('training: {}'.format (normallogstore))
    output ('target:   {}'.format (args.target))
    output ('results saved in: {}'.format (savefile))
    output ('')
    output ('normal entries:      {} / {} = {}'.format (
        n_normal, n_total, n_normal / n_total))
    output ('attack entries:      {} / {} = {}'.format (
        n_attack, n_total, n_attack / n_total))
    output ('correct predictions: {} / {} = {}'.format (
        n_correct, n_total, n_correct / n_total))
    output ('true positives:  {}'.format (n_true_pos))
    output ('true negatives:  {}'.format (n_true_neg))
    output ('false positives: {}'.format (n_false_pos))
    output ('false negatives: {}'.format (n_false_neg))
    output ('precision: {}'.format (precision))
    output ('recall:    {}'.format (recall))
    output ('F-score:   {}'.format (f_score))
