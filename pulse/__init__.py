#!/usr/bin/env python
# encoding: utf-8

from pulse.kNN import reduce_dim, train_kNN
from pulse.read_data import *


def train(path):
    data = train_data(path)
    labels, samples = zip_data(data)
    new, m = reduce_dim(labels, samples)
    data_set = to_data_set(labels, new)
    print train_kNN(data_set)
