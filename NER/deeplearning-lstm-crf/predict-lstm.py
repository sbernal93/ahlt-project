#!/usr/bin/env python3

import sys
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

import matplotlib.pyplot as plt

import pickle
import numpy as np
import pandas as pd

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
    #return xseq,toks

def load_data(fi):
    xseq = []
    yseq = []

    for line in fi:
        line = line.strip('\n')
        if not line:
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]
        xseq.append(item)

        # Append the label to the label sequence.
        toks.append([fields[0],fields[1],fields[2],fields[3]])
    return xseq, yseq

if __name__ == '__main__':

    model = pickle.load(open(sys.argv[1], "rb"))
    word2idx = pickle.load(open(sys.argv[2], "rb"))
    utags = pickle.load(open(sys.argv[3], "rb"))
    max_len = 75

    for xseq,toks in instances_pred(sys.stdin):
        #x = pd.DataFrame(xseq)
        x_test_sent = [[word2idx.get(t[0],0) for t in xseq]]
        x_test_sent = pad_sequences(maxlen=max_len, sequences=x_test_sent, padding="post", value=0)

        prediction = model.predict(np.array([x_test_sent[0]]))
        inside = False;
        for k in range(0,len(prediction)) :
            y = utags[prediction[k]]
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
