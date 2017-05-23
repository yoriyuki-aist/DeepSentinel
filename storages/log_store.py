import pandas as pd
import numpy as np
import itertools
import operator
from pathlib import Path
import re
from tqdm import tqdm

# labels for real swat log
# simulated log does not have labels
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
    def __init__(self, filename, normal=None, simulated=False, filenames=None):
        if simulated:
            self.filename = filename
            logs = []
            for path in tqdm(filenames):
                with open(path) as f:
                    log = []
                    for line in tqdm(f):
                        log.append(re.findall('[0-9.]+', line))
                    log = np.array(log, dtype=np.float32)
                    logs.append(log)
            concat = np.concatenate(logs)
            if normal is None:
                self.means = np.mean(concat, axis=0)
                self.stds = np.std(concat, axis=0)
                logs = [(log - self.means) / self.stds for log in logs]
            else:
                logs = [(log - normal.means) / normal.stds for log in logs]
            self.values_seq = np.stack(logs, axis=-1)
            self.position_units = 0
            self.value_units = logs[0].shape[1]

        else:
            log = pd.read_excel(filename, header=[0, 1])
            log.index = self.log.index.to_datetime()
            self.filename = Path(filename).stem

            positions = self.log[discrete_labels]
            self.positions_seq = positions.values

            #FIXME use a common code with for the simulated log
            if normal is None:
                self.means = [log.mean(label) for label in value_labels]
                self.stds = [log.std(label) for label in value_labels]
                zscore = lambda x: (x - self.means[x.name]) / self.stds[x.name]
                values = log[value_labels].apply(zscore)
            else:
                zscore = lambda x: (x - normal.means[x.name]) / normal.stds[x.name]
                values = log[value_labels].apply(zscore)
            values.fillna(0)

            #FIXME Need chunking here
            self.values_seq = values.values

            self.position_units = len(discrete_labels)
            self.value_units = len(value_labels)
