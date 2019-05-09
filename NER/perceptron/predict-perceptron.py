#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer

def instances_pred(fi):
    xseq = []
    toks = []

    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line means the end of a sentence.
            # Return accumulated sequences, and reinitialize.
            yield xseq, toks
            xseq = []
            toks = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]
        xseq.append(item)

        # Append the label to the label sequence.
        toks.append([fields[0],fields[1],fields[2],fields[3]])

def encode(data):
    x = np.asarray(data)
    le = LabelEncoder()
    for k in range(x.shape[1]):
        x[:,k] = le.fit_transform(x[:,k])
    return x.astype(int)

if __name__ == '__main__':

    model = pickle.load(open(sys.argv[1], "rb"))
    v = pickle.load(open(sys.argv[2], "rb"))

    for xseq,toks in instances_pred(sys.stdin):
        x = pd.DataFrame(xseq)
        x = v.transform(x.to_dict('records'))
        prediction = model.predict(x)
        inside = False;
        for k in range(0,len(prediction)) :
            y = prediction[k]
            (sid, form, offS, offE) = toks[k]

            if (y[0]=="B") :
                entity_form = form
                entity_start = offS
                entity_end = offE
                entity_type = y[2:]
                inside = True
            elif (y[0]=="I" and inside) :
                entity_form += " "+form
                entity_end = offE
            elif (y[0]=="O" and inside) :
                print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|")
                inside = False

        if inside : print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|")
