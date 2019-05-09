#!/usr/bin/env python3

import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
import pickle
import numpy as np
import pandas as pd

label_replace = {'O': 0, 'B-drug': 1, 'I-drug': 2, 'B-drug_n': 3,
'I-drug_n': 4, 'B-group': 5, 'I-group': 6, 'B-brand': 7, 'I-brand': 8}

def instances(fi):
    xseq = []
    yseq = []

    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line means the end of a sentence.
            # Return accumulated sequences, and reinitialize.
            #yield xseq, yseq
            #xseq = []
            #yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]
        xseq.append(item)

        # Append the label to the label sequence.
        yseq.append(fields[4])
    return xseq, yseq

def encode(data):
    x = np.asarray(data)
    le = LabelEncoder()
    for k in range(x.shape[1]):
        x[:,k] = le.fit_transform(x[:,k])
    return x.astype(int)

def encode_tags(test):
    y = np.asarray(test)
    for k in range(y.shape[0]):
        y[k] = label_replace[y[k]]
    return y.astype(int)




if __name__ == '__main__':

    v = DictVectorizer(sparse=False)
    (xtrain, ytrain) = instances(sys.stdin)
    x = pd.DataFrame(xtrain)
    y = pd.DataFrame(ytrain).values[:,0]

    x = v.fit_transform(x.to_dict('records'))
    classes = np.unique(y)
    classes = classes.tolist()

    per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
    per = per.partial_fit(x, y, classes)

    #model = pipeline.fit(x.to_dict('records'), y)

    with open(sys.argv[1], 'wb') as f:
        pickle.dump(per, f)

    with open(sys.argv[2], 'wb') as f:
        pickle.dump(v, f)
