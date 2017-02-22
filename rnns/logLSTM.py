import numpy as np
import chainer
from chainer import Variable, functions as F, links as L

class LogLSTM(chainer.Chain):
    def __init__(self, index_units, num_units, value_units, n_units):
        super(LogLSTM, self).__init__(
            embed_label = L.EmbedID(label_units, label_units),
            embed_note = L.EmbedID(note_units, n_units),
            l1 = L.LSTM(label_units + n_units + 2, n_units),
            l2 = L.LSTM(n_units, n_units),
            output = L.Linear(n_units, 5 * command_units),
        )
        self.command_units = command_units

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    #ネットワークの学習、lossを計算
    def __call__(self, data):
        loss = 0

        for cur, nt in data:
            loss += self.eval(cur, nt, train=True)

        return loss

    def eval(self, current_data, next_data, volatile='off', train=False):
        labels = self.embed_label(Variable(current_data[0], volatile=volatile))
        notes = self.embed_note(Variable(current_data[1], volatile=volatile))
        vals = Variable(current_data[3], volatile=volatile)
        l0 = F.concat((labels, notes, vals), axis=1)
        h1 = F.dropout(self.l1(l0), train=train)
        h2 = F.dropout(self.l2(h1), train=train)
        y = self.output(h2)

        commands_nt = Variable(next_data[2], volatile=volatile)
        vals_nt = Variable(next_data[3], volatile=volatile)

        y_commands, y_vals = F.split_axis(y, [self.command_units], 1)
        loss = F.softmax_cross_entropy(y_commands, commands_nt,  use_cudnn=False)

        mean = F.get_item(F.dstack([F.select_item(y_vals, 4*commands_nt), F.select_item(y_vals, 4*commands_nt+1)]), 0)
        logvar2 = F.get_item(F.dstack([F.select_item(y_vals, 4*commands_nt+2), F.select_item(y_vals, 4*commands_nt+3)]), 0)
        loss += F.gaussian_nll(vals_nt, mean, logvar2)

        return loss
