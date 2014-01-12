#!/usr/bin/env python
# encoding: utf-8

from mvpa2.clfs.svm import RbfCSVMC
from mvpa2.measures.base import CrossValidation

import numpy as np

def train_SVM(data_set, partitioner):
    clf = RbfCSVMC(0.8)
    clf.set_postproc(None)
    cv = CrossValidation(clf, partitioner)
    cv_results = cv(data_set)
    return np.mean(cv_results)
