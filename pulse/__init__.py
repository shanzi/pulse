#!/usr/bin/env python
# encoding: utf-8

from pulse.knn import reduce_dim, cv_kNN
from pulse.svm import cv_SVM, train_SVM
from pulse.read_data import *
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.datasets import dataset_wizard
import os

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



def run_test(train_path, test_path, output_path):
    print "Reading training data"
    train_datas = train_data(train_path)
    labels, samples = zip_data(train_datas)
    train_data_set = to_data_set(labels, samples)
    print "Training model"
    clf = train_SVM(train_data_set)
    test_datas = test_data(test_path)
    print "Reading test data"
    test_datas = test_data(test_path)
    test_features = []
    file_names = []
    for file_, data in test_datas:
        feature = test_data_feature(data)
        test_features.append(feature)
        file_names.append(file_)
    array = np.array(test_features)
    predictions = clf.predict(array)
    write_test_result(test_path, output_path, file_names, predictions)
