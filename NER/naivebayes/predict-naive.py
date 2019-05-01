#!/usr/bin/env python3

import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

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


    for xseq,toks in instances_pred(sys.stdin):
        prediction = model.predict(encode(xseq))
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
