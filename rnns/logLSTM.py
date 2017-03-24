import numpy as np
import chainer
from chainer import Variable, functions as F, links as L
from storages import log_store

class LogLSTM(chainer.Chain):
    def __init__(self, lstm_num, position_num, position_units, value_units, n_units):

        position_input_layer = chainer.ChainList()
        for i in range(position_units - 1):
            position_input_layer.add_link(L.Bilinear(position_num, n_units, n_units))

        lstm_stack = chainer.ChainList()
        for i in range(lstm_num):
            lstm_stack.add_link(L.LSTM(n_units, n_units))

        output_layers = chainer.ChainList()
        for i in range(position_units):
            output_layers.add_link(L.Bilinear(position_num, n_units, n_units + position_num))
        for i in range(value_units - 1):
            output_layers.add_link(L.Bilinear(1, n_units, n_units + 2))

        super(LogLSTM, self).__init__(
            position1 = L.EmbedID(position_num, n_units),
            position_input_layer = position_input_layer,
            input_layer = L.BiLinear(n_units, value_units, n_units),
            lstms = lstm_stack,
            output_position1 = L.Linear(n_units, position_num),
            output_layers = output_layers,
            output_last_value = L.Linear(n_units, 2)
        )
        self.position_num = position_num
        self.lstm_num = lstm_num
        self.position_units = position_units
        self.value_units = value_units

    def reset_state(self):
        for i in range(lstm_num):
            self.lstms.reset_state()

    def __call__(self, data):
        loss = 0

        for cur, nt in data:
            loss += self.eval(cur, nt, volatile='off', train=True)

        return loss

    def eval(self, cur, nt, volatile='on', train=False):
        xp = self.xp
        positions_cur, values_cur = cur
        positions_nt, values_nt = nt

        p1 = self.position1(Variable(xp.array(positions_cur[0].T, xp.int)))


        identity_matrix = Variable(xp.ones(self.position_num, dtype=xp.float32))
        embed_positions = F.embed_id(positions_cur[1:], identity_matrix)

        inpt = self.xp.array(current.T, self.xp.float32)
        outpt = self.xp.array(nt.T, self.xp.float32)
        vector = self.input(Variable(inpt, volatile=volatile))

        h1 = F.dropout(self.l1(vector), train=train)
        h2 = F.dropout(self.l2(h1), train=train)
        y = self.output(h2)
        y_index, mean, logvar2 = F.split_axis(y, [self.index_units, self.index_units + self.value_units], 1)
        y_indices = F.split_axis(y_index, list(range(3, self.index_units, 3)), 1)

        next_var = Variable(outpt, volatile=volatile)
        next_index, next_value = F.split_axis(next_var, [self.index_units], 1)
        next_indices = F.split_axis(next_index, list(range(3, self.index_units, 3)), 1)
        loss = 0.0
        for y_i, next_i in zip(y_indices, next_indices):
            loss += F.sum(F.matmul(-F.log_softmax(y_i), F.transpose(next_i)))
        loss += F.gaussian_nll(next_value, mean, logvar2)

        return loss
