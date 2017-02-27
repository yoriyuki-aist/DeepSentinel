import numpy as np
import chainer
from chainer import Variable, functions as F, links as L

class LogLSTM(chainer.Chain):
    def __init__(self, index_units, value_units, n_units):
        super(LogLSTM, self).__init__(
            index_inputs = [L.Linear(3, n_units) for i in range(index_units)]
            value_input = L.Linear(value_units, n_units)
            l1 = L.LSTM(n_units, n_units),
            l2 = L.LSTM(n_units, n_units),
            output = L.Linear(n_units, 3 * index_units + 2 * value_units),
        )
        self.index_units = index_units
        self.value_units = value_units

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, data):
        loss = 0

        for cur, nt in data:
            loss += self.eval(cur, nt, train=True)

        return loss

    def eval(self, current_data, next_data, volatile='off', train=False):
        current_indices = current_data[0]
        next_indices = next_data[0]
        current_values = current_data[1]
        next_values = next_data[1]

        index_vector = sum([index_inputs[i](current_indices[i], volatile=volatile) for i in range(self.index_units)])
        value_vector = value_input(curret_values, volatile=volatile)

        h1 = F.dropout(self.l1(index_vector + value_vector), train=train)
        h2 = F.dropout(self.l2(h1), train=train)
        y = self.output(h2)

        loss = 0.0
        for i in range(self.index_units):
            next_indix_var = Variable(next_indices[i], volatile=volatile)
            loss += F.softmax_cross_entropy(y[3*i:3*i+3], next_index_var,  use_cudnn=False)

        for i in range(self.value_unit):
            next_value_var = Variable(next_values[i], volatile=volatile)
            mean = F.select_item(y, 3 * self.index_units + 2 * i)
            logvar2 = F.select_item(y, 3 * self.index_units + 2 * i + 1)
            loss += F.gaussian_nll(next_value_var, mean, logvar2)

        return loss
