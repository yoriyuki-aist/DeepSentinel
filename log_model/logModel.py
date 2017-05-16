import pickle
import itertools
import functools
import chainer
import math
import numpy as np
from chainer import cuda, optimizers, serializers
from rnns import logLSTM
from rnns.logLSTM import LogLSTM
from tqdm import tqdm
import sys

def batch_seq(seq, chunk_num):
    seq = seq[0:len(seq) // chunk_num * chunk_num]
    chunked = np.split(seq, chunk_num)
    return np.stack(chunked, axis=-1)

class LogModel:
    def __init__(self, log_store, lstm_num = 1, n_units=1000, tr_sq_ln=100, gpu=-1, directory='', logLSTM_file=None, optimizer_file=None):
        self.log_store = log_store
        self.n_units= n_units
        self.tr_sq_ln = tr_sq_ln
        self.gpu = gpu
        self.dir = directory
        self.current_epoch = 0
        self.chunk_num = 10
        self.lstm_num = lstm_num

        self.model = LogLSTM(lstm_num, 3, log_store.position_units, log_store.value_units, self.n_units)
        if not logLSTM_file is None:
            serializers.load_npz(logLSTM_file, self.model)
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.model.to_gpu()

        self.optimizer = optimizers.Adam()
        if not optimizer_file is None:
            serializers.load_npz(optimizer_file, self.optimizer)

    def train(self, epoch):
        if self.current_epoch >= epoch:
            pass
        else:
            model = self.model
            optimizer = self.optimizers

            for j in tqdm(range(self.current_epoch+1, epoch+1)):
                model.reset_state()

                ps_seq = batch_seq(self.log_store.positions_seq, self.chunk_num)
                vs_seq = batch_seq(self.log_store.values_seq, self.chunk_num)
                ps_cur = ps_seq[:-1]
                vs_cur = vs_seq[:-1]
                cur = zip(ps_cur, vs_cur)
                ps_nt = ps_seq[1:]
                vs_nt = vs_seq[1:]
                nt = zip(ps_nt, vs_nt)
                data = zip(cur, nt)

                loss_sum = 0
                for k in tqdm(range(0, len(ps_seq) - 1, self.tr_sq_ln)):
                    model.cleargrads()
                    data_seq = list(itertools.islice(data, self.tr_sq_ln))
                    loss = model(data_seq)
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()
                    loss_sum += loss.data

                with open(self.dir+"{}-stat-{}-{}.csv".format(self.log_store.filename, self.lstm_num, self.n_units),'a') as statfile:
                    print(j, ',',  loss_sum, file=statfile)
                self.current_epoch = j
                self.save()

    def _eval(self, seq):
        sys.exit('logModel._eval is no longer implemented.')

    def eval(self, ps_seq, vs_seq):
        ps_cur = ps_seq[:-1]
        vs_cur = vs_seq[:-1]
        cur = zip(ps_cur, vs_cur)
        ps_nt = ps_seq[1:]
        vs_nt = vs_seq[1:]
        nt = zip(ps_nt, vs_nt)
        data = tqdm(zip(cur, nt))

        self.model.reset_state()
        return (self.model.eval(cur, nt) for cur, nt in data)

    def save(self):
        filename = self.dir+"{}-model-{}-{}-{}-lstms.npz".format(self.log_store.filename, self.lstm_num, self.n_units, self.current_epoch)
            serializers.save_npz(filename, self.model)
        filename = self.dir+"{}-model-{}-{}-{}-optimizer.npz".format(self.log_store.filename, self.lstm_num, self.n_units, self.current_epoch)
            serializers.save_npz(filename, self.optimizer)
