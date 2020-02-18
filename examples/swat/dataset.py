from pathlib import Path

import pandas as pd

from deep_sentinel.utils import to_absolute


class SWaTData(object):
    datetime_format = " %d/%m/%Y %I:%M:%S %p"

    all_continuous_columns = {
        # P1
        'FIT101', 'LIT101',
        # P2
        'AIT201', 'AIT202', 'AIT203', 'FIT201',
        # P3
        'DPIT301', 'FIT301', 'LIT301',
        # P4
        'AIT401', 'AIT402', 'FIT401', 'LIT401',
        # P5
        'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'PIT501', 'PIT502', 'PIT503',
        # P6
        'FIT601'
    }

    all_discrete_columns = {
        # P1
        'MV101', 'P101', 'P102',
        # P2
        'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
        # P3
        'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
        # P4
        'P401', 'P402', 'P403', 'P404', 'UV401',
        # P5
        'P501', 'P502',
        # P6
        'P601', 'P602', 'P603'
    }

    class_label = 'Normal/Attack'
    normal_label_column = 'Normal'
    attack_label_column = 'Attack'

    # These columns always have the same fixed value
    fixed_on_normal = {'P102', 'P201', 'P202', "P204", "P205", 'P206', 'P401', 'P403', 'P404', 'P502', 'P601', 'P603'}
    fixed_on_attack = {'P202', "P204", "P205", 'P401', 'P404', 'P502', 'P601', 'P603'}

    continuous_columns = list(all_continuous_columns.difference(fixed_on_normal))
    discrete_columns = list(all_discrete_columns.difference(fixed_on_normal))

    def __init__(self, dataset_file: 'Path'):
        self.path = to_absolute(dataset_file)
        self.df = None

    def read(self) -> 'pd.DataFrame':
        df = pd.read_excel(str(self.path), skiprows=1, parse_dates=False)

        # Some column titles contain white space
        df.rename(columns=lambda x: x.strip(), inplace=True)

        # Convert object type to timestamp index.
        df['Timestamp'] = pd.to_datetime(df["Timestamp"], format=self.datetime_format)
        df.set_index('Timestamp', inplace=True)

        # Convert discrete value columns as category data.
        df[self.discrete_columns] = df[self.discrete_columns].astype('category')
        label_data = pd.get_dummies(df[self.class_label]).astype('category')
        # Drop Normal/Attack column
        df.drop(self.class_label, axis=1, inplace=True)
        df[self.normal_label_column] = label_data[self.normal_label_column]
        try:
            df[self.attack_label_column] = label_data[self.attack_label_column]
        except KeyError:
            # There is no attack data on normal.
            df[self.attack_label_column] = 0
            df[self.attack_label_column] = df[self.attack_label_column].astype('category')
        self.df = df
        return self.df[
            [
                *self.continuous_columns,
                *self.discrete_columns,
                self.attack_label_column,
                self.normal_label_column,
            ]
        ]
