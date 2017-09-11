import argparse
import numpy as np
import pickle
from sklearn import svm
from pathlib import Path
import sys
import time
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import expon, gamma
from functools import reduce
# from log_store import LogStore

if __name__ == '__main__':
    start_time = time.time ()

    default_kernel='rbf'
    default_nu_scale=0.01
    default_gamma_scale=0.01
    default_window=4

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('training', metavar='F',
                        help='Pickled training data (without output/ or extension)')
    parser.add_argument('--nu', '-n', type=float, default=default_nu_scale,
                        help='Scale of upper bound on training error rate [default: {}]'.format (default_nu_scale))
    parser.add_argument('--gamma', '-g', type=float, default=default_gamma_scale,
                        help='Scale of kernel coefficient [default: {}]'.format (default_gamma_scale))
    parser.add_argument('--kernel', '-k', type=str, default=default_kernel,
                        help='Kernel (rbf, poly, or sigmoid) [default: {}]'.format (default_kernel))
    parser.add_argument('--window', '-w', type=int, default=default_window,
                        help='How many entries to mash into one sample [default: {}]'.format (default_window))
    parser.add_argument('--debug', metavar='N', type=int, default=0,
                        help='Level of debugging output')
    parser.add_argument('test',
                        help='Pickled test data')

    args = parser.parse_args()

    if not Path('output').exists():
        os.mkdir('output')
    else:
        if not Path('output').is_dir():
            sys.exit('output is not a directory.')

    logname = Path(args.training).stem
    logstore = (Path('output') / logname).with_suffix('.pickle')

    print("loading training...")
    if logstore.exists():
        with logstore.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        print('cannot open log file {} - did you run preprocess.py?'.format (logstore),
                  file=sys.stderr)
        sys.exit(1)

    testlogname = Path (args.test).stem
    testlogstore = (Path ('output') / testlogname).with_suffix ('.pickle')
    print ('test data =', testlogstore)
    if testlogstore.exists():
        with testlogstore.open(mode='rb') as testlogstorefile:
            test_log_store = pickle.load(testlogstorefile)
    else:
        sys.exit(1)

    print("preprocessing...")

    pp_start_time = time.time ()

    values = np.concatenate (log_store.train_v_seqs)
    positions = np.concatenate (log_store.train_p_seqs)
    data = np.concatenate ([values, positions], axis=1)
    if args.debug >= 2:
        print (data)
    window = [data[i:-(args.window-i)] for i in range (args.window)]
    data = np.concatenate (window, axis=1)
    print ('training data size :', data.shape)

    test_labels = np.concatenate (test_log_store.train_l_seqs)
    test_values = np.concatenate (test_log_store.train_v_seqs)
    test_positions = np.concatenate (test_log_store.train_p_seqs)
    test_data = np.concatenate ([test_values, test_positions], axis=1)

    is_normal = (test_labels == 'Normal')

    if args.debug >= 1:
        print ('test_data :', test_data.shape)
    if args.debug >= 2:
        print (test_data)
    test_window = [test_data[i:-(args.window-i)] for i in range (args.window)]
    is_normal_window = [is_normal[i:-(args.window-i)] for i in range (args.window)]
    test_data = np.concatenate (test_window, axis=1)
    is_normal = reduce (np.logical_and, is_normal_window)
    print ('test data size :', test_data.shape)

    if args.debug >= 4:
        print ('window\n', window)
        print ('data\n', data)

    pp_end_time = time.time ()

    print("start learning...")

    params = ParameterSampler({'nu': expon(scale=args.nu), 'gamma': expon(scale=args.gamma)}, n_iter=1000)

    for param in params:
        nu = param['nu']
        gamma = param['gamma']

        print ('nu =', nu)
        print ('gamma =', gamma)
        if args.debug >= 1:
            print ('kernel =', args.kernel)
            print ('data :', data.shape)

        learn_start_time = time.time ()

        model = svm.OneClassSVM (nu=nu, kernel=args.kernel, gamma=gamma)
        model.fit (data)

        learn_end_time = time.time ()

        file_stem = '{}-svm-w{}-n{}-g{}-{}'.format (logname, args.window, nu, gamma, args.kernel)

        modelfilename = file_stem + '.pickle'
        modelfile = (Path ('output') / modelfilename)
        with modelfile.open ('wb') as f:
            pickle.dump (model, f)

        print('start evaluating...')
        eval_start_time = time.time ()

        pred_test = model.predict (test_data)

        eval_end_time = time.time ()

        predfile = (Path('output') / (file_stem + '-results.pickle'))
        with predfile.open ('wb') as f:
            pickle.dump (pred_test, f)

        print ('compiling results...')

        stat_start_time = time.time ()

        # NB: prediction values are -1 if attack, +1 if normal
        pred_normal = (pred_test == 1)
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

        log = (Path('output') / (file_stem + '.log'))
        f = open (log, 'w')
        def output (x):
            print (x, file=f, flush=True)

        output ('model:    {}'.format (modelfile))
        output ('training: {}'.format (logname))
        output ('testing:   {}'.format (testlogname))
        output ('results saved in: {}'.format (predfile))
        output ('')
        output ('preprocessing time:    {}'.format (pp_end_time - pp_start_time))
        output ('training time:       {}'.format (learn_end_time - learn_start_time))
        output ('evaluation time:       {}'.format (eval_end_time - eval_start_time))
        output ('stat compilation time: {}'.format (stat_end_time - stat_start_time))
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
        f.close()
