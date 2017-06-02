import pandas as pd
import numpy as np
import itertools
import operator
from pathlib import Path

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

def batch_seq(seq, chunk_num):
    seq = seq[0:len(seq) // chunk_num * chunk_num]
    chunked = np.split(seq, chunk_num)
    return np.stack(chunked, axis=-1)

class LogStore:
    def __init__(self, filename, normal=None):
        self.position_units = len(discrete_labels)
        self.value_units = len(value_labels)
        self.filename = Path(filename).stem

        self.log = pd.read_excel(filename, header=[0, 1])
        self.log.index = pd.to_datetime(self.log.index)
        self.log['F'] = 1 #the row is filled
        new_index = pd.date_range(start=self.log.index[0], end=self.log.index[-1], freq='S', normalize=True)
        self.log = self.log.reindex(new_index)
        positions = self.log[discrete_labels]
        filled = self.log[['F']]
        positions = positions.fillna(0)
        filled = filled.fillna(0)

        if normal is None:
            zscore = lambda x: (x - x.mean()) / x.std()
            values = self.log[value_labels].apply(zscore)
        else:
            zscore = lambda x: (x - normal.log[x.name].mean()) / normal.log[x.name].std()
            values = self.log[value_labels].apply(zscore)
        values = values.fillna(0)


        if normal is None:
            grouped = values.groupby(values.index.map(lambda x: x.date()))
            values_seqs = grouped.apply(lambda x: x.values)
            rest = list(values_seqs.iloc[:-1])
            lastday = values_seqs.iloc[-1]
            self.train_v_seq = np.stack(rest, axis=-1)
            self.test_v_seq = np.stack([lastday], axis=-1)

            grouped = positions.groupby(positions.index.map(lambda x: x.date()))
            positions_seqs = grouped.apply(lambda x: x.values)
            rest = list(positions_seqs.iloc[:-1])
            lastday = positions_seqs.iloc[-1]
            self.train_p_seq = np.stack(rest, axis=-1)
            self.test_p_seq = np.stack([lastday], axis=-1)

            grouped = filled.groupby(filled.index.map(lambda x: x.date()))
            f_seqs = grouped.apply(lambda x: x.values)
            rest = list(f_seqs.iloc[:-1])
            lastday = positions_seqs.iloc[-1]
            self.train_f_seq = np.stack(rest, axis=-1)
            self.test_f_seq = np.stack([lastday], axis=-1)
        else:
            self.test_v_seq = np.stack([values.values], axis=-1)
            self.test_p_seq = np.stack([positions.values], axis=-1)
            self.test_f_seq = np.stack([filled.values], axis=-1)
