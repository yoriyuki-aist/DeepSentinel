import pandas as pd
import numpy as np
import itertools

class LogStore:
    def __init__(self, filename):
        log = pd.read_excel(filename)

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
