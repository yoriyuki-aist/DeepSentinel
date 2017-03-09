import pickle
import itertools
import functools
import chainer
import math
import numpy as np
from chainer import cuda, optimizers
from rnns import logLSTM
from rnns.logLSTM import LogLSTM
from tqdm import tqdm

class LogModel:
    def __init__(self, log_store, n_units=1000, tr_sq_ln=100, gpu=-1, directory=''):
        self.log_store = log_store
        self.n_units= n_units
        self.tr_sq_ln = tr_sq_ln
        self.gpu = gpu
        self.dir = directory
        self.current_epoch = 0
        self.chunk_num = 10
        self.train_seq = log_store.training_seq(self.chunk_num)
        self.test_seq = log_store.test_seq(self.chunk_num)

        self.model = LogLSTM(log_store.index_units, log_store.value_units, self.n_units)

    def train(self, epoch):
        if self.current_epoch >= epoch:
            pass
        else:
            model = self.model
            if self.gpu >= 0:
                cuda.get_device(self.gpu).use()
                model.to_gpu()

            optimizer = optimizers.Adam()
            optimizer.setup(model)

            for j in tqdm(range(self.current_epoch+1, epoch+1)):
                model.reset_state()
                cur, nt = itertools.tee(self.train_seq)
                nt = itertools.islice(nt, 1, None)
                data = zip(cur, nt)
                for k in tqdm(range(0, len(self.train_seq), self.tr_sq_ln)):
                    model.cleargrads()
                    data_seq = list(itertools.islice(data, self.tr_sq_ln))
                    loss = model(data_seq)
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()

                self.current_epoch = j
                self.save()
                model.reset_state()
                loss = self.eval(self.test_seq, self.log_store.filename)
                with open(self.dir+"stat-{}-{}.csv".format(self.log_store.filename, self.n_units),'a') as statfile:
                    print(j, ',',  2**loss, file=statfile)

    def _eval(self, seq):
        self.model.reset_state()
        cur, nt = itertools.tee(seq)
        nt = itertools.islice(nt, 1, None)
        data = zip(cur, nt)
        return (self.model.eval(cur, nt, volatile='on').data for cur, nt in data)

    def eval(self, seq, filename):
        count = 0
        sum_loss = 0
        with open(self.dir+"{}-{}-{}.csv".format(filename, self.n_units, self.current_epoch), 'w') as f:
            for outlier_factor in self._eval(tqdm(seq)):
                sum_loss += outlier_factor
                print(outlier_factor, file=f)
                count += 1
        return sum_loss / len(seq)

    def save(self):
        with open(self.dir+"{}-model-{}-{}.pickle".format(self.log_store.filename, self.n_units, self.current_epoch), 'wb') as f:
            pickle.dump(self, f)
