#!/usr/bin/env python3

import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

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

    gnb = GaussianNB()

    vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

    #xtrain = []
    #ytrain = []
    (xtrain, ytrain) = instances(sys.stdin)
    #for xseq, yseq in instances(sys.stdin):
    #    xtrain.append(xseq)
    #    ytrain.append(yseq)
    xtrain = encode(xtrain)
    #ytrain = encode_tags(ytrain)

    pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=20)),
                     ('clf', GaussianNB())])

    model = gnb.fit(np.array(xtrain), np.array(ytrain))

    with open(sys.argv[1], 'wb') as f:
        pickle.dump(model, f)

#https://towardsdatascience.com/multi-class-text-classification-with-sklearn-and-nltk-in-python-a-software-engineering-use-case-779d4a28ba5
#https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
