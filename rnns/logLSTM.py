import numpy as np
import chainer
from chainer import Variable, functions as F, links as L
from storages import log_store

class LogLSTM(chainer.Chain):
    def __init__(self, lstm_num, position_num, position_units, value_units, n_units, activation):

        lstm_stack = chainer.ChainList()
        for i in range(lstm_num):
            lstm_stack.add_link(L.LSTM(n_units, n_units))

        output_pos_layers = chainer.ChainList()
        for i in range(position_units - 1):
            output_pos_layers.add_link(L.Bilinear(position_num, n_units, n_units))

        output_pos_mid_layers = chainer.ChainList()
        for i in range(position_units - 1):
            output_pos_mid_layers.add_link(L.Linear(n_units, n_units + position_num))

        output_pos_mid_layers = chainer.ChainList()
        for i in range(position_units - 1):
            output_pos_mid_layers

        output_val_layers = chainer.ChainList()
        for i in range(value_units - 1):
            output_val_layers.add_link(L.Bilinear(1, n_units, n_units))

        output_val_mid_layers = chainer.ChainList()
        for i in range(position_units - 1):
            output_val_mid_layers.add_link(L.Linear(n_units, n_units + 2))

        super(LogLSTM, self).__init__(
            input_layer = L.Linear(position_num * position_units + value_units, n_units),
            lstms = lstm_stack,
            output_pos1 = L.Linear(n_units, position_num),
            output_pos_layers = output_pos_layers,
            output_pos_mid_layers = output_pos_mid_layers,
            output_lastpos = L.Bilinear(position_num, n_units, n_units),
            output_val_layers = output_val_layers,
            output_val_mid_layers = output_val_mid_layers,
            output_last_value = L.Linear(n_units, 2)
        )
        self.position_num = position_num
        self.lstm_num = lstm_num
        self.position_units = position_units
        self.value_units = value_units
        self.n_units = n_units
        if activation == 'sigmoid':
            self.activate = F.sigmoid
        elif activation == 'relu':
            self.activate = F.relu
        else:
            pass

    def reset_state(self):
        for i in range(self.lstm_num):
            self.lstms[i].reset_state()

    def __call__(self, data, dropout=True):
        loss = 0.0

        for cur, nt in data:
            loss += self.eval(cur, nt, volatile='off', train=dropout)

        return loss

    def eval(self, cur, nt, volatile='on', train=False):
        if volatile == 'on':
            with chainer.no_backprop_mode():
                if train == False:
                    with chainer.using_config('train', False):
                        loss = self._eval(cur, nt, train)
                else:
                    loss = self._eval(cur, nt, train)
        else:
            if train == False:
                with chainer.using_config('train', False):
                    loss = self._eval(cur, nt, train)
            else:
                loss = self._eval(cur, nt, train)

        return loss

    def _eval(self, cur, nt, train=False):
        xp = self.xp
        positions_cur, values_cur = cur
        positions_nt, values_nt = nt

        ps_in = Variable(xp.array(positions_cur.T, dtype=xp.int32))
        identity_matrix = Variable(xp.identity(self.position_num, dtype=xp.float32))
        ps_in_dm = F.embed_id(ps_in, identity_matrix)
        ps_in_dm = F.reshape(ps_in_dm, (ps_in_dm.shape[0], self.position_num * self.position_units))

        vs_in = Variable(xp.array(values_cur.T, dtype=xp.float32))
        h = F.dropout(self.input_layer(F.concat((ps_in_dm,vs_in))))
        for i in range(self.lstm_num):
            h = F.dropout(self.lstms[i](h))

        y = h
        y_pos = []
        y_pos.append(self.output_pos1(y))
        ps_true = Variable(xp.array(positions_nt.T, dtype=xp.int32))
        ps_true_dm = F.embed_id(ps_true, identity_matrix)
        ps_true_dm = F.reshape(ps_true_dm, (ps_true_dm.shape[0], self.position_num * self.position_units))
        ps_true_dm = F.split_axis(ps_true_dm, self.position_units, 1)
        for i in range(self.position_units - 1):
            z = F.dropout(self.activate(self.output_pos_layers[i](ps_true_dm[i], y)))
            z = self.output_pos_mid_layers[i](z)
            y, p_out = F.split_axis(z, [self.n_units], 1)
            y = F.dropout(self.activate(y))
            y_pos.append(p_out)

        p_true_dm = ps_true_dm[-1]
        y_val = []
        y = F.dropout(self.activate(self.output_lastpos(p_true_dm, y)))
        vs_true = Variable(xp.array(values_nt.T, dtype=xp.float32))
        vs_true = F.split_axis(vs_true, self.value_units, 1)
        for i in range(self.value_units - 1):
            y = self.output_val_mid_layers[i](y)
            y, val_out = F.split_axis(y, [self.n_units], 1)
            y_val.append(val_out)
            y = F.dropout(self.activate(self.output_val_layers[i](vs_true[i], y)))
        val_out = self.output_last_value(y)
        y_val.append(val_out)

        loss = 0.0
        ps_true = F.split_axis(ps_true, self.position_units, 1)
        for i in range(self.position_units):
            with chainer.using_config('use_cudnn', 'never'):
                loss += F.softmax_cross_entropy(y_pos[i], F.flatten(ps_true[i]))

        for i in range(self.value_units):
            val_out, = F.split_axis(y_val[i], 1, 1)
            loss += F.gaussian_nll(F.flatten(vs_true[i]), val_out[:,  0], val_out[:, 1])

        return loss
