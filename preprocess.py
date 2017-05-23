#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pickle
from pathlib import Path
import glob
import os
import sys
from storages import log_store
from storages.log_store import LogStore
from log_model import logModel
from log_model.logModel import LogModel

if __name__ == '__main__':

    #コマンドライン引数
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--simulated', type=bool, default=False, help='true if the logfiles are simlated')
    parser.add_argument('logfile', metavar='F', help='log file')
    parser.add_argument('logfiles', metavar='FS', nargs='*', help='log files')

    args = parser.parse_args()

    if not Path('output').exists():
        os.mkdir('output')
    else:
        if not Path('output').is_dir():
            sys.exit('output is not a directory.')

    logname = Path(args.logfile).stem
    logstore = (Path('output') / logname).with_suffix('.pickle')

    print("loading log file...")
    if args.simulated:
        log_store = LogStore(logname, filenames=args.logfiles, simulated=True)
    else:
        log_store = LogStore(args.logfile)

    with logstore.open(mode='wb') as f:
        pickle.dump(log_store, f)
