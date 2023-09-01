import csv
import numpy as np
import sys
import pandas as pd
import itertools
import math
import time

from sklearn import svm, linear_model, neighbors
from sklearn import tree, ensemble
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold

import networkx as nx
import random
import numbers

def generatePairs(ddi_df, embedding_df):

    drugs = set(ddi_df.Drug1.unique())
    drugs = drugs.union(ddi_df.Drug2.unique())
    drugs = drugs.intersection(embedding_df.Drug.unique())

    ddiKnown = set([tuple(x) for x in  ddi_df[['Drug1','Drug2']].values])

    pairs = list()
    classes = list()

    for dr1,dr2 in itertools.combinations(sorted(drugs),2):
        if dr1 == dr2: continue

        if (dr1,dr2)  in ddiKnown or  (dr2,dr1)  in ddiKnown:
            cls=1
        else:
            cls=0

        pairs.append((dr1,dr2))
        classes.append(cls)

    pairs = np.array(pairs)
    classes = np.array(classes)

    return pairs, classes

def balance_data(pairs, classes, n_proportion):
    classes = np.array(classes)
    pairs = np.array(pairs)

    indices_true = np.where(classes == 1)[0]
    indices_false = np.where(classes == 0)[0]

    np.random.shuffle(indices_false)
    indices = indices_false[:(n_proportion*indices_true.shape[0])]
    print ("+/-:", len(indices_true), len(indices), len(indices_false))
    pairs = np.concatenate((pairs[indices_true], pairs[indices]), axis=0)
    classes = np.concatenate((classes[indices_true], classes[indices]), axis=0)

    return pairs, classes

