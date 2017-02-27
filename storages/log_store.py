import pandas as pd
import numpy as np
import itertools
import operator

value_labels = [('P1', 'FIT101'),
('P1', 'LIT101'),
('P2', 'AIT201'),
('P2', 'AIT202'),
('P2', 'AIT203'),
('P2', 'FIT201'),
('P3', 'DPIT301'),
('P3', 'FIT301'),
('P3', 'LIT301'),
('P4', 'AIT401'),
('P4', 'AIT402'),
('P4', 'FIT401'),
('P4', 'LIT401'),
('P5', 'AIT501'),
('P5', 'AIT502'),
('P5', 'AIT503'),
('P5', 'AIT504'),
('P5', 'FIT501'),
('P5', 'FIT502'),
('P5', 'FIT503'),
('P5', 'FIT504'),
('P5', 'PIT501'),
('P5', 'PIT502'),
('P5', 'PIT503'),
('P6', 'FIT601') ]

discrete_labels = [
('P1', 'MV101'),
('P1', 'P101'),
('P1', 'P102'),
('P2', 'MV201'),
('P2', 'P201'),
('P2', 'P202'),
('P2', 'P203'),
('P2', 'P204'),
('P2', 'P205'),
('P2', 'P206'),
('P3', 'MV301'),
('P3', 'MV302'),
('P3', 'MV303'),
('P3', 'MV304'),
('P3', 'P301'),
('P3', 'P302'),
('P4', 'P401'),
('P4', 'P402'),
('P4', 'P403'),
('P4', 'P404'),
('P4', 'UV401'),
('P5', 'P501'),
('P5', 'P502'),
('P6', 'P601'),
('P6', 'P602'),
('P6', 'P603'),]


class LogStore:
    def __init__(self, filename):
        self.log = pd.read_excel(filename, header=[0, 1])
        self.log.index = self.log.index.to_datetime()
        self.last_day = self.log.tail(1).index[0].date()

        zscore = lambda x: (x - x.mean()) / x.std()
        self.values = self.log[value_labels].apply(zscore)

        self.indices = self.log[discrete_labels]
        self.indices2num = None

    def eval_seq(self, indices2num):
        self.log[('num', '')] = self.indices.apply(lambda x: indices2num[tuple(x)], axis=1)
        self.indices = self.log[discrete_labels + [('num', '')]]

        value_shape = self.values.values.shape
        indices_shape = self.indices.values.shape

        values_seq = self.values.values.reshape(value_shape[0], 1, value_shape[1])
        indices_seq = self.indices.values.reshape(indices_shape[0], 1, indices_shape[1])
        return (values_seq, indices_seq)

    def training_seq(self):
        if self.indices2num == None:
            index_set = set(map(tuple, self.log[discrete_labels].values.tolist()))
            index_list= sorted(list(index_set))
            self.indices2num = dict(zip(index_list, itertools.count(1)))
            self.maxnum = max(self.indices2num.items(), key=operator.itemgetter(1))[1]

        self.log[('num', '')] = self.indices.apply(lambda x: self.indices2num[tuple(x)], axis=1)
        self.indices = self.log[discrete_labels + [('num', '')]]

        values = self.values[self.values.index.date != self.last_day]
        indices = self.indices[self.values.index.date != self.last_day]

        grouped_by_day = values.groupby(lambda x: x.date)
        dates = grouped_by_day.groups.keys()
        values_seq = []
        indices_seq = []
        for date in dates:
            values_by_day = values.loc[pd.Series(grouped_by_day.groups[date])]
            indices_by_day = indices.loc[pd.Series(grouped_by_day.groups[date])]
            values_seq.append(values_by_day.values)
            indices_seq.append(indices_by_day.values)
        return (np.array(values_seq).T, np.array(indices_seq).T)

    def test_seq(self):
        if self.indices2num == None:
            self.training_seq()

        self.log[('num', '')] = self.indices.apply(lambda x: self.indices2num.get(tuple(x), 0), axis=1)
        self.indices = self.log[discrete_labels + [('num', '')]]

        values = self.values[self.values.index.date == self.last_day]
        indices = self.indices[self.values.index.date == self.last_day]

        value_shape = values.values.shape
        indices_shape = indices.values.shape

        values_seq = values.values.reshape(value_shape[0], 1, value_shape[1])
        indices_seq = indices.values.reshape(indices_shape[0], 1, indices_shape[1])
        return (values_seq, indices_seq)
