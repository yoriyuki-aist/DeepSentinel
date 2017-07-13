import argparse
import numpy as np
import pickle
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

    if args.debug >= 4:
        print ('cur\n', cur)
        print ('nxt\n', nxt)
        print ('data\n', data)

    print("start evaluating...")

    if args.debug >= 1:
        print ('data :', data.shape)
    if args.debug >= 4:
        print ('data\n', data)

    pred_eval = model.predict (data)
    n_error_eval = pred_eval[pred_eval == -1].size
    print ('{}:\t{}/{} = {}'.format (args.target, n_error_eval, pred_eval.size, n_error_eval / pred_eval.size))
