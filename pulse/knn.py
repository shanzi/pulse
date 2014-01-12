#!/usr/bin/env python
# encoding: utf-8

from mvpa2.clfs.knn import kNN
from mvpa2.measures.base import CrossValidation
from pulse.lda import lda
import numpy as np

def reduce_dim(labels, samples):
    return lda(np.array(samples), np.array(labels), 2)

def cv_kNN(data_set, partitioner):
    clf = kNN(12)
    clf.set_postproc(None)
    cv = CrossValidation(clf, partitioner)
    cv_results = cv(data_set)
    return np.mean(cv_results)

