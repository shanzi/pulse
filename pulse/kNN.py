#!/usr/bin/env python
# encoding: utf-8

from mvpa2.clfs.knn import kNN
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import HalfPartitioner
from pulse.lda import lda
import numpy as np

def reduce_dim(labels, samples):
    return lda(np.array(samples), np.array(labels), 2)

def train_kNN(data_set):
    clf = kNN()
    clf.set_postproc(None)
    hpart = HalfPartitioner(attr='runtype')
    cv = CrossValidation(clf, hpart)
    cv_results = cv(data_set)
    return np.mean(cv_results)

