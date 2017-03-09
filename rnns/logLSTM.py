import numpy as np
import chainer
from chainer import Variable, functions as F, links as L
from storages import log_store

class LogLSTM(chainer.Chain):
    def __init__(self, index_units, value_units, n_units):
        super(LogLSTM, self).__init__(
            input = L.Linear(index_units + value_units, n_units),
            l1 = L.LSTM(n_units, n_units),
            l2 = L.LSTM(n_units, n_units),
            output = L.Linear(n_units, index_units + 2 * value_units),
        )
        self.index_units = index_units
        self.value_units = value_units

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, data):
        loss = 0

        for cur, nt in data:
            loss += self.eval(cur, nt, volatile='off', train=True)

        return loss

    def eval(self, current, nt, volatile='on', train=False):
        inpt = self.xp.array(current.T, self.xp.float32)
        outpt = self.xp.array(nt.T, self.xp.float32)
        vector = self.input(Variable(inpt, volatile=volatile))

        h1 = F.dropout(self.l1(vector), train=train)
        h2 = F.dropout(self.l2(h1), train=train)
        y = self.output(h2)
        y_index, mean, logvar2 = F.split_axis(y, [self.index_units, self.index_units + self.value_units], 1)

        next_var = Variable(outpt, volatile=volatile)
        next_index, next_value = F.split_axis(next_var, [self.index_units], 1)

        loss = F.sum(F.matmul(-F.log_softmax(y_index), F.transpose(next_index)))
        loss += F.gaussian_nll(next_value, mean, logvar2)

        return loss
