import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from pathlib import Path
import sys
import time
# from log_store import LogStore

if __name__ == '__main__':
    start_time = time.time ()

    default_kernel='rbf'
    default_nu=0.1
    default_gamma=0.1
    default_window=1

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('logfile', metavar='F',
                        help='Pickled log file (without output/ or extension)')
    parser.add_argument('--nu', '-n', type=float, default=default_nu,
                        help='Upper bound on training error rate [default: {}]'.format (default_nu))
    parser.add_argument('--gamma', '-g', type=float, default=default_gamma,
                        help='Kernel coefficient [default: {}]'.format (default_gamma))
    parser.add_argument('--kernel', '-k', type=str, default=default_kernel,
                        help='Kernel (rbf, poly, or sigmoid) [default: {}]'.format (default_kernel))
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

    logname = Path(args.logfile).stem
    logstore = (Path('output') / logname).with_suffix('.pickle')

    print("loading log file...")
    if logstore.exists():
        with logstore.open(mode='rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        print('cannot open log file {} - did you run preprocess.py?'.format (logstore),
                  file=sys.stderr)
        sys.exit(1)

    print("preprocessing...")

    pp_start_time = time.time ()

    values = np.concatenate (log_store.train_v_seqs)
    positions = np.concatenate (log_store.train_p_seqs)
    data = np.concatenate ([values, positions], axis=1)

    print ('data size :', data.shape)
    if args.debug >= 2:
        print (data)
    window = [data[i:-(args.window-i)] for i in range (args.window)]
    data = np.concatenate (window, axis=1)

    if args.debug >= 4:
        print ('window\n', window)
        print ('data\n', data)

    pp_end_time = time.time ()

    print("start learning...")

    if args.debug >= 1:
        print ('nu =', args.nu)
        print ('kernel =', args.kernel)
        print ('gamma =', args.gamma)
        print ('data :', data.shape)

    learn_start_time = time.time ()

    model = svm.OneClassSVM (nu=args.nu, kernel=args.kernel, gamma=args.gamma)
    model.fit (data)

    learn_end_time = time.time ()

    file_stem = '{}-svm-w{}-n{}-g{}-{}'.format (logname, args.window, args.nu, args.gamma, args.kernel)

    savefilename = file_stem + '.pickle'
    savefile = (Path ('output') / savefilename)
    with savefile.open ('wb') as f:
        pickle.dump (model, f)

    pred_start_time = time.time ()

    pred_train = model.predict (data)
    n_error_train = pred_train[pred_train == -1].size
    print ('error train: {}/{} = {}'.format (n_error_train, pred_train.size, n_error_train / pred_train.size))

    pred_end_time = time.time ()
    end_time = pred_end_time

    with (Path ('output') / (file_stem + '.time')).open ('a') as f:
        print ('preprocessing time: {}'.format (pp_end_time - pp_start_time), file=f)
        print ('learning time:      {}'.format (learn_end_time - learn_start_time), file=f)
        print ('scoring time:       {}'.format (pred_end_time - pred_start_time), file=f)
        print ('total time:         {}'.format (end_time - start_time), file=f)
