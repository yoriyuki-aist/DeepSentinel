import pandas as pd
import numpy as np
import itertools
import operator
import pickle
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

def chunked(seq, chunk_num):
    seq = seq[0:len(seq) // chunk_num * chunk_num]
    return np.split(seq, chunk_num)

class LogStore:
    def __init__(self, filename, normal_filename=None):
        self.position_units = len(discrete_labels)
        self.value_units = len(value_labels)


        self.log = pd.read_excel(filename, header=[0, 1])
        self.log.index = pd.to_datetime(self.log.index)
        self.filename = Path(filename).stem
        #Assuming the log is continous

        normal = None
        if not normal_filename is None:
            print("loading normal log file...")
            normallogname = Path(normal_filename).stem
            normallogstore = (Path('output') / normallogname).with_suffix('.pickle')
            if normallogstore.exists():
                with normallogstore.open(mode='rb') as normallogstorefile:
                    normal = pickle.load(normallogstorefile)

        if normal is None:
            zscore = lambda x: (x - x.mean()) / x.std()
            values = self.log[value_labels].apply(zscore)
        else:
            zscore = lambda x: (x - normal.log[x.name].mean()) / normal.log[x.name].std()
            values = self.log[value_labels].apply(zscore)
        values.fillna(0)

        positions = self.log[discrete_labels]

        if normal is None:
            i_seqs = chunked(self.log.index.values, 10)
            v_seqs = chunked(values.values, 10)
            p_seqs = chunked(positions.values, 10)
        else:
            i_seqs = chunked(self.log.index.values, 1)
            v_seqs = chunked(values.values, 1)
            p_seqs = chunked(positions.values, 1)

        self.train_i_seqs = i_seqs[:-1]
        self.test_i_seq = i_seqs[-1]
        self.train_v_seqs = v_seqs[:-1]
        self.test_v_seq = v_seqs[-1]
        self.train_p_seqs = p_seqs[:-1]
        self.test_p_seq = p_seqs[-1]
