#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pickle
import os
import glob
import os.path
from storages import log_store
from storages.log_store import LogStore
from log_model import logModel
from log_model.logModel import LogModel

if __name__ == '__main__':

    #コマンドライン引数
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_units', '-n', type=int, default=2000,
                        help='Number of n_units')
    parser.add_argument('--iter', '-i', type=int, default=10,
                    help='Number of epochs')
    parser.add_argument('--cont', '-c', default=False,
                help='Continue to learn')
    parser.add_argument('logfile', metavar='F')
    parser.add_argument('chunk', metavar='C', type=int)

    args = parser.parse_args()


    if not os.path.exists('output'):
        os.mkdir('output')

    print("loading log file...")
    if args.cont and os.path.isfile('output/log_store.pickle'):
        with open('output/log_store.pickle', 'rb') as logstorefile:
            log_store = pickle.load(logstorefile)
    else:
        log_store = LogStore(args.logfile)

    with open('output/log_store.pickle', 'wb') as f:
        pickle.dump(log_store, f)
    print("start learning...")

    if args.cont:
        for epoch in range(args.iter, 0, -1):
            log_model_name = "output/log_model-{}-{}-{}.pickle".format(args.chunk, args.n_units, epoch)
            if os.path.isfile(log_model_name):
                break

        if os.path.isfile(log_model_name):
            with open(log_model_name, 'rb') as modelfile:
                log_model = pickle.load(modelfile)
        else:
            log_model = LogModel(log_store, args.chunk, args.n_units, gpu=args.gpu, directory='output/')
    else:
        log_model = LogModel(log_store, args.chunk, args.n_units, gpu=args.gpu, directory='output/')

    if log_model.current_epoch == args.iter:
        log_model.eval()
    else:
        log_model.train(args.iter)
