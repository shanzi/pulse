#!/usr/bin/env python
# encoding: utf-8

from pulse.kNN import reduce_dim, cv_kNN
from pulse.svm import cv_SVM
from pulse.read_data import *
from mvpa2.generators.partition import NFoldPartitioner

def cross_validate(path):
    data = train_data(path)
    labels, samples = zip_data(data)
    partitioner = NFoldPartitioner(attr="runtype")
    data_set = to_data_set(labels, samples)
    accurracy_svm= 1-cv_SVM(data_set, partitioner)
    print "Accuracy SVM: %.2f" % accurracy_svm
    new, m = reduce_dim(labels, samples)
    data_set = to_data_set(labels, new)
    accurracy_knn = 1-cv_kNN(data_set, partitioner)
    print "Accuracy kNN: %.2f" % accurracy_knn


def run_test(path):
    pass
