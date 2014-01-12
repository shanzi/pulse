#!/usr/bin/env python
# encoding: utf-8

import os
from scipy.io import loadmat, savemat
import numpy as np
from mvpa2.datasets import dataset_wizard

DATA_NAME = 'Beats'
DATA_FIELD = 'beatData'
PEAK_FIELD = 'rPeak'
TEST_DATA_FIELD = 'data'

def zip_data(data_list):
    return zip(*data_list)

def to_data_set(labels, samples):
    ds = dataset_wizard(samples, targets=labels)
    len_ = len(labels)
    a = len_/5
    runtype = a*range(5)
    if len(runtype)<len_: runtype += [1]*(len_-len(runtype))
    ds.sa['runtype']=runtype
    return ds


def train_data(dir_root):
    abs_dir_root = os.path.abspath(dir_root)

    def walk_train_dir(train_data, dir_name, files):
        label = os.path.split(dir_name)[-1].strip()
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
                    for beat in beats:
                        peak = np.amax(beat,0)
                        trouph = np.amin(beat,0)
                        train_data.append((label, peak-trouph))

    train_data = []
    os.path.walk(abs_dir_root, walk_train_dir, train_data)
    return train_data

def test_data(dir_root):
    abs_dir_root = os.path.abspath(dir_root)
    def walk_test_dir(test_data, dir_name, files):
        for file_ in files:
            name, ext = os.path.splitext(file_)
            if ext == '.mat':
                path = os.path.join(dir_name, file_)
                mat_contents = loadmat(path)
                mat_data = mat_contents[TEST_DATA_FIELD]
                test_data.append((file_, mat_data))

    test_data = []
    os.path.walk(abs_dir_root, walk_test_dir, test_data)
    return test_data

def test_data_feature(raw):
    samples = raw[:,1:]
    max_ = np.amax(samples, 0)
    min_ = np.amin(samples, 0)
    return (max_-min_).tolist()


def write_test_result(test_path, output_path, file_names, predictions):
    abs_test = os.path.abspath(test_path)
    abs_output = os.path.abspath(output_path)
    for file_, type_ in zip(file_names, predictions):
        test_m = loadmat(os.path.join(test_path, file_))
        test_m['label'] = type_
        savemat(os.path.join(output_path, file_), test_m)
