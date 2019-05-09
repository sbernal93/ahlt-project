#!/usr/bin/env python3

import sys
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd

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


if __name__ == '__main__':


    v = DictVectorizer(sparse=False)
    (xtrain, ytrain) = instances(sys.stdin)
    x = pd.DataFrame(xtrain)
    y = pd.DataFrame(ytrain).values[:,0]

    pipeline = Pipeline([('vect', v),
                    ('chi',  SelectKBest(chi2, k=10000)),
                    ('clf', ComplementNB())])
    model = pipeline.fit(x.to_dict('records'), y)

    with open(sys.argv[1], 'wb') as f:
        pickle.dump(model, f)

#https://towardsdatascience.com/multi-class-text-classification-with-sklearn-and-nltk-in-python-a-software-engineering-use-case-779d4a28ba5
#https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958
#https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af
#https://www.datacamp.com/community/tutorials/categorical-data
