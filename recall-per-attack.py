import argparse
import pickle
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Precision-Recall-Fmeasure')
    parser.add_argument('attacklist', metavar='AttackList', help='List of attacks')
    parser.add_argument('verdictfile', metavar='VerdictFile', help='Verdicts')
    parser.add_argument('label', metavar='ColumnLabel', help='Label of the column')
    parser.add_argument('outputfile', metavar='Outputfile', help='output file')
    args = parser.parse_args()

    attacklist = pd.read_excel(args.attacklist)
    data = pd.DataFrame(attacklist['Start Time'])

    def endtime(row):
        starttime = row['Start Time']
        if row['End Time'] == '':
            return np.nan
        time = pd.to_timedelta(str(row['End Time']))
        endtime1 = pd.to_datetime(starttime.date()) + time
        endtime2 = endtime1 + pd.to_timedelta('1d')
        if endtime1 >= starttime:
            return endtime1
        else:
            return endtime2

    data['End Time'] = attacklist.apply(endtime, axis=1)
    print(data)

    f = open(args.verdictfile, 'rb')
    verdicts = pickle.load(f)
    print(verdicts)
    f.close()
    timestamps = pd.to_datetime(verdicts.index)

    def recall(row):
        starttime = row['Start Time']
        endtime = row['End Time']
        if pd.isnull(endtime) or pd.isnull(starttime):
            return np.nan
        sliced = verdicts[(timestamps >= starttime) & (timestamps <= endtime)]
        num = len(sliced)
        if num == 0:
            return np.nan
        true_positive_num = sliced.sum()
        return true_positive_num / num

    data[args.label] = data.apply(recall, axis=1)
    print(data)
    data.to_excel(args.outputfile)
