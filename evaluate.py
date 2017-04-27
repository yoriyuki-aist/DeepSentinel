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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute outlier factors and evaluate them')
    parser.add_argument('model', metavar='Model file', help='Model file')
    parser.add_argument('target', metavar='Target', help='Log to be analyzed')
    parser.add_argument('output', metavar='Output', help='Ooutput')

    args = parser.parse_args()


    if not Path(args.model).exists():
        sys.exit('model does not exists.')
    else:
        with Path(args.model).open(mode='rb') as modelfile:
            print("loading model file...")
            log_model = pickle.load(modelfile)

    print('loading target log file')
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
        log_store = LogStore(args.logfile)

    print("start evaluating...")
    scores = np.fromiter(log_model.eval(log_store.position_seq, log_store.values_seq), np.float32)
    log = logstore.log
    log['score'] = pd.Series(scores).shift(1)
    log = log.dropna(axis=0)

    normal_dummy = pd.get_dummies(log['Normal/Attack'])['Normal']
    log['Normal'] = normal_dummy
    log['Attack'] = 1 - normal_dummy

    log = log.sort_value(by='score', ascending=False)

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
    k = log['score'].loc[max_f_index]
    f = log['f_value'].loc[max_f_index]
    p = log['precision'].loc[max_f_index]
    r = log['recall'].loc[max_f_index]
    print("Threshold: {}, Precision: {}, Recall: {}, F value: {}".format(k, p, r, f))

    judgements = pd.cut(log['score'].values, [log['score'].min(), k, log['score'].max()+1], right=False, labels=['N', 'A'])
    log['judge'] = judgements

    print('AUC: {}'.format(metrics.auc(fp.values, recall.values)))
    log.to_excel(args.output)
