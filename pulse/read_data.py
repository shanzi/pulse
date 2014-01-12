#!/usr/bin/env python
# encoding: utf-8

import os
from scipy.io import loadmat
import numpy as np
from mvpa2.datasets import dataset_wizard

DATA_NAME = 'Beats'
DATA_FIELD = 'beatData'
PEAK_FIELD = 'rPeak'

def zip_data(data_list):
    return zip(*data_list)

def to_data_set(labels, samples):
    ds = dataset_wizard(samples, targets=labels)
    len_ = len(labels)
    a = len_/2
    runtype = a*[0, 1]
    if len(runtype)<len_: runtype += [1]*(len_-len(runtype))
    ds.sa['runtype']=runtype
    return ds


def train_data(dir_root):
    abs_dir_root = os.path.abspath(dir_root)

    def walk_train_dir(train_data, dir_name, files):
        dirname = os.path.split(dir_name)[-1].strip()
        label = False if dirname=='normal' else True
        for file_ in files:
            name, ext = os.path.splitext(file_)
            if ext == '.mat':
                path = os.path.join(dir_name, file_)
                mat_contents = loadmat(path)
                mat_data = mat_contents.get(DATA_NAME)
                if mat_data!=None:
                    val = mat_data[0,0]
                    beats = val[DATA_FIELD]
                    peak = val[PEAK_FIELD]
                    beats = np.squeeze(beats[0])
                    peak = np.squeeze(peak[0])
                    for i, beat in enumerate(beats):
                        idx = peak[i]
                        if idx > 0:
                            sliced = beat[idx]
                            train_data.append((label, sliced))

    train_data = []
    os.path.walk(abs_dir_root, walk_train_dir, train_data)
    return train_data

