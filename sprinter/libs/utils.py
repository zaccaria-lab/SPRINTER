import sys, os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import datetime
import subprocess as sp
from multiprocessing import Pool, Manager
import warnings
import contextlib
import time

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from sklearn.cluster import mean_shift
from statsmodels.distributions.empirical_distribution import ECDF

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from math import floor, ceil
from multiprocessing.sharedctypes import Value
from unittest import skip



get_consecutive = (lambda a : zip(a, np.append(a[1:], None)))

norm_mean = (lambda array : array / array.mean())

get_index_safe = (lambda A, idx : A.loc[idx] if idx is not None else np.nan)
get_prev_notna = (lambda A : pd.Series(index=A.index, data=(get_index_safe(A, A.iloc[:idx].last_valid_index()) for idx in range(A.size))))
get_next_notna = (lambda A : pd.Series(index=A.index, data=(get_index_safe(A, A.iloc[idx+1:].first_valid_index()) for idx in range(A.size))))


class ProgressBar:

    def __init__(self, total, length, counter=0, verbose=False, decimals=1, fill=chr(9608), prefix = 'Progress:', suffix = 'Complete'):
        self.total = total
        self.length = length
        self.decimals = decimals
        self.fill = fill
        self.prefix = prefix
        self.suffix = suffix
        self.counter = counter
        self.verbose = verbose

    def progress(self, advance=True, msg=""):
        self.progress_unlocked(advance, msg)
        return True

    def progress_unlocked(self, advance, msg):
        flush = sys.stderr.flush
        write = sys.stderr.write
        if advance:
            self.counter += 1
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.counter / float(self.total)))
        filledLength = int(self.length * self.counter // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        rewind = '\x1b[2K\r'
        result = '%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix)
        msg = '[{:%Y-%b-%d %H:%M:%S}]'.format(datetime.datetime.now()) + msg
        if not self.verbose:
            toprint = rewind + result + " [%s]" % (msg)
        else:
            toprint = rewind + msg + "\n" + result
        write(toprint)
        flush()
        if self.counter == self.total:
            write("\n")
            flush()


def log(msg, level='STEP', lock=None):
    timestamp = '{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if level == "STEP":
        if lock is None:
            sys.stderr.write("{}{}[{}]{}{}\n".format(bcolors.BOLD, bcolors.HEADER, timestamp, msg, bcolors.ENDC))
        else:
            with lock: sys.stderr.write("{}{}[{}]{}{}\n".format(bcolors.BOLD, bcolors.HEADER, timestamp, msg, bcolors.ENDC))
    elif level == "INFO":
        if lock is None:
            sys.stderr.write("{}[{}]{}{}\n".format(bcolors.OKGREEN, timestamp, msg, bcolors.ENDC))
        else:
            with lock: sys.stderr.write("{}[{}]{}{}\n".format(bcolors.OKGREEN, timestamp, msg, bcolors.ENDC))
    elif level == "WARN":
        if lock is None:
            sys.stderr.write("{}[{}]{}{}\n".format(bcolors.WARNING, timestamp, msg, bcolors.ENDC))
        else:
            with lock: sys.stderr.write("{}[{}]{}{}\n".format(bcolors.WARNING, timestamp, msg, bcolors.ENDC))
    elif level == "PROGRESS":
        if lock is None:
            sys.stderr.write("{}{}[{}]{}{}\n".format(bcolors.UNDERLINE, bcolors.BBLUE, timestamp, msg, bcolors.ENDC))
        else:
            with lock: sys.stderr.write("{}[{}]{}{}\n".format(bcolors.BBLUE, timestamp, msg, bcolors.ENDC))
    elif level == "ERROR":
        if lock is None:
            sys.stderr.write("{}[{}]{}{}\n".format(bcolors.FAIL, timestamp, msg, bcolors.ENDC))
        else:
            with lock: sys.stderr.write("{}[{}]{}{}\n".format(bcolors.FAIL, timestamp, msg, bcolors.ENDC))
    else:
        if lock is None:
            sys.stderr.write("{}\n".format(msg))
        else:
            with lock: sys.stderr.write("{}\n".format(msg))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    BBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

