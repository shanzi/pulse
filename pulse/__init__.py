#!/usr/bin/env python
# encoding: utf-8

from pulse.kNN import reduce_dim, train_kNN
from pulse.svm import train_SVM
from pulse.read_data import *
from mvpa2.generators.partition import NFoldPartitioner

def train(path):
    data = train_data(path)
    labels, samples = zip_data(data)
    partitioner = NFoldPartitioner(attr="runtype")
    #new, m = reduce_dim(labels, samples)
    #data_set = to_data_set(labels, new)
    #print train_kNN(data_set)
    data_set = to_data_set(labels, samples)
    print 1-train_SVM(data_set, partitioner)
