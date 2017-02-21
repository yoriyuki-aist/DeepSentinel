import pandas as pd
import numpy as np
import itertools

class LogStore:
    def __init__(self, filename):
        log = pd.read_excel(filename, header=[0, 1])
        log.index = log.index.to_datetime()
        self.last_day = log.tail(1).index[0].date()

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
        zscore = lambda x: (x - x.mean()) / x.std()
        self.values = log[value_labels].apply(zscore)

        discreate_labels = [
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
        self.indices = log[dicreate_labels]
        index_set = set()
        for indicies in self.indices.values.tolist():
            set.add(tuple(indices))
        num = 0
        indices2num = {}
        for indices in index_set:
            indices2num[num] = indices
            num += 1




    def eva_seq(self):
        values = self.values
        inices = self.indices

        grouped_by_day = values.groupby(lambda x: x.date)
        dates = grouped_by_day.groups.key()
        values_seq = []
        indices_seq = []
        for date in dates:
            values_by_day = values.loc[pd.Series(grouped_by_day.groups[date])]
            indices_by_day = indices.loc[pd.Series(grouped_by_day.groups[date])]
            values_seq.append(values_by_day.values)
            indices_seq.append(indices_by_day.values)
        return (nq.array(values_seq).T, nq.array(indices_seq).T)

    def training_seq(self):
        values = self.values[self.values.index.date != self.last_day]
        inices = self.indices[self.values.index.date != self.last_day]

        grouped_by_day = values.groupby(lambda x: x.date)
        dates = grouped_by_day.groups.key()
        values_seq = []
        indices_seq = []
        for date in dates:
            values_by_day = values.loc[pd.Series(grouped_by_day.groups[date])]
            indices_by_day = indices.loc[pd.Series(grouped_by_day.groups[date])]
            values_seq.append(values_by_day.values)
            indices_seq.append(indices_by_day.values)
        return (nq.array(values_seq).T, nq.array(indices_seq).T)

    def test_seq(self):
        values = self.values[self.values.index.date == self.last_day]
        inices = self.indices[self.values.index.date == self.last_day]

        grouped_by_day = values.groupby(lambda x: x.date)
        dates = grouped_by_day.groups.key()
        values_seq = []
        indices_seq = []
        for date in dates:
            values_by_day = values.loc[pd.Series(grouped_by_day.groups[date])]
            indices_by_day = indices.loc[pd.Series(grouped_by_day.groups[date])]
            values_seq.append(values_by_day.values)
            indices_seq.append(indices_by_day.values)
        return (nq.array(values_seq).T, nq.array(indices_seq).T)
