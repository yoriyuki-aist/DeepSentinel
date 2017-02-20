import pandas as pd
import numpy as np
import time
import itertools
from datetime import datetime, time
from sklearn import preprocessing

unused_command = {'fridge1_status',
'fridge2_status',
'location_omzn',
'sentiment',
'target_14873_status',
'target_15001_status',
'target_15002_status',
'target_15006_status',
'target_15070_status',
'target_15071_status',
'target_15084_status',
'target_15086_status',
'target_15087_status',
'target_15088_status',
'target_15089_status',
'target_15090_status',
'target_15093_status',
'target_15096_status',
'target_15097_status',
'target_15098_status',
'target_15099_status',
'target_15102_status',
'target_15103_status',
'target_15104_status',
'target_15105_status',
'target_15106_status',
'target_15107_status',
'target_15108_status',
'target_15109_status',
'target_15110_status',
'target_192\.168\.68\.15_status',
'target_192\.168\.68\.21_status',
'target_192\.168\.68\.23_status',
'target_192\.168\.68\.25_status',
'target_192\.168\.68\.26_status',
'target_192\.168\.68\.27_status',
'target_192\.168\.68\.28_status',
'target_192\.168\.68\.30_status',
'target_192\.168\.68\.32_status',
'target_192\.168\.68\.33_status',
'target_192\.168\.68\.36_status',
'target_192\.168\.68\.38_status',
'target_31480_status',
'target_B4:18:D1:4F:41:A3_status',
'target_e2c56db5dffb48d2b060d0f5a71096e0_status',
'target_e2c56db5dffb48d2b060d0f5a71096e1_status'}

def chunking(data, size):
    yield list(itertools.islice(data, size))

class LogStore:
    def __init__(self, filename, chunk_num=10):
        self.chunk_num = chunk_num
        label2index = {}
        note2index = {}
        command2index = {}
        log = pd.read_csv(filename)

        #Data cleansing
        log = log[log['label'].isin(unused_command).map(lambda b: not b)]
        log = log.dropna(subset=['label'])

        #command indexing
        labels = set()
        notes = set()
        command_0 = set()
        command_num = set()
        command_str = set()
        command_2 = set()
        for command, note, value in log[['label', 'note', 'value']].itertuples(index=False, name=None):
            if pd.isnull(note) and pd.isnull(value):
                labels.add(command)
                command_0.add(command)
            elif pd.isnull(value):
                labels.add(command)
                notes.add(note)
                command_str.add((command, note))
            elif pd.isnull(note):
                labels.add(command)
                command_num.add(command)
            else:
                labels.add(command)
                labels.add(note)
                command_2.add((command, note))

        label_index = 0
        for label in sorted(list(labels)):
            label2index[label] = label_index
            label_index += 1
        self.label_num = label_index

        note_index = 1 # 0 is used for nan
        for note in sorted(list(notes)):
            note2index[note] = note_index
            note_index += 1
        self.note_num = note_index

        index = 0
        for command in sorted(list(command_num)):
            command2index[(command, 0)] = index
            index += 1
        for command, note in sorted(list(command_2)):
            command2index[(command, note, 0)] = index
        for command in sorted(list(command_0)):
            command2index[command] = index
            index += 1
        for command, note in sorted(list(command_str)):
            command2index[(command, note)] = index
            index += 1
        self.command_num = index

        #time delta
        grouped = log.groupby('label')
        log['timedelta'] = pd.to_datetime(log['timestamp'].shift(0)) -   pd.to_datetime(log['timestamp'].shift())

        zscore = lambda x: (x - x.mean()) / x.std()
        values = log[['label', 'value', 'timedelta']].groupby('label').transform(zscore).fillna(0)

        value_seq = values.values

        IDs = []
        label_seq = []
        note_seq = []
        command_seq = []
        for ID, label, value, note in log[['id', 'label', 'value', 'note']].itertuples(index=False, name=None):
            IDs.append(ID)
            if pd.isnull(note) and pd.isnull(value):
                command_index = command2index[label]
            elif pd.isnull(note):
                command_index = command2index[(label, 0)]
            elif pd.isnull(np.nan):
                command_index = command2index[(label, note)]
            else:
                command_index = command2index[(label, note, 0)]
            command_seq.append(command_index)

            label_seq.append(label2index[label])

            if pd.isnull(note):
                note_index = 0
            else:
                note_index = note2index[note]
            note_seq.append(note_index)

        self.chunk_size = len(IDs) // chunk_num

        self.ID_seqs = np.array_split(np.array(IDs), chunk_num)
        self.label_seqs = np.array_split(np.array(label_seq), chunk_num)
        self.note_seqs = np.array_split(np.array(note_seq), chunk_num)
        self.command_seqs = np.array_split(np.array(command_seq), chunk_num)
        self.value_seqs = np.array_split(np.array(value_seq), chunk_num)
