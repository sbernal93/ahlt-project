#!/usr/bin/env python3

import sys
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras_contrib.layers import CRF

import matplotlib.pyplot as plt

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
            yield xseq, yseq
            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]
        xseq.append(item)

        # Append the label to the label sequence.
        yseq.append(fields[4])
    #return xseq, yseq

def load_data(fi):
    xtrain = []
    ytrain = []

    for line in fi:
        line = line.strip('\n')
        if not line:
            continue
        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]
        xtrain.append(item)

        # Append the label to the label sequence.
        ytrain.append(fields[4])
    return xtrain, ytrain



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

if __name__ == '__main__':

    sentences = []
    tags = []

    fi = open(sys.argv[1])

    # Read sentences from STDIN, and append them to the trainer.
    for xseq, yseq in instances(fi):
        sentences.append(xseq)
        tags.append(yseq)
    fi.close()
    fi = open(sys.argv[1])
    (xtrain, ytrain) = load_data(fi)
    x = pd.DataFrame(xtrain)
    y = pd.DataFrame(ytrain).values[:,0]

    max_len = 75
    words = list(set(x[0].values))
    words.append("ENDPAD")
    n_words = len(words)
    utags = list(set(y))
    n_tags = len(utags)

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(utags)}

    docs = [[w[0] for w in s] for s in sentences]
    #docs.append("UNKNOWN")

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)

    Y = [[tag2idx[t] for t in tag ]for tag in tags]
    Y = pad_sequences(maxlen=max_len, sequences=Y, padding="post", value=tag2idx["O"])
    Y = [to_categorical(i, num_classes=n_tags) for i in Y]

    max_len_char = 10
    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)
    print(n_chars)

    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))


    word_in = Input(shape=(max_len,))
    emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                 input_length=max_len, mask_zero=True)(word_in)

    # input and embeddings for characters
    char_in = Input(shape=(max_len, max_len_char,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                       input_length=max_len_char, mask_zero=True))(char_in)
    # character LSTM to get word encodings by characters
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                            recurrent_dropout=0.5))(emb_char)

    # main LSTM
    x = concatenate([emb_word, char_enc])
    #x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.6))(x)
    model = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    model = Model([word_in, char_in], out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    history = model.fit([X, np.array(X_char).reshape((len(X_char), max_len, max_len_char))],
                        np.array(Y), batch_size=32, epochs=5,
                    validation_split=0.1, verbose=1)

    hist = pd.DataFrame(history.history)

    out = open(sys.argv[3], "w+")
    for xseq,toks in instances_pred(open(sys.argv[2])):
        #x = pd.DataFrame(xseq)
        x_test_sent = [[word2idx.get(t[0],0) for t in xseq]]
        x_test_sent = pad_sequences(maxlen=max_len, sequences=x_test_sent, padding="post", value=n_words-1)

        sentence = [t[0] for t in xseq]
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)

        #prediction = model.predict(np.array([x_test_sent[0]]))[0]
        #model.predict([X_word_te,
        #                    np.array(X_char_te).reshape((len(X_char_te),
        #                                                 max_len, max_len_char))])
        x_test_sent = np.array([x_test_sent[0]])
        prediction = model.predict([x_test_sent, np.array(sent_seq).reshape((len(x_test_sent), max_len, max_len_char))])[0]
        prediction = np.argmax(prediction, axis=-1)
        inside = False;
        for k in range(0,len(toks)) :
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
                #print(sid, entity_start+"-"+entity_end, entity_form, entity_type, sep="|")
                out.write(sid + "|" + entity_start+"-"+entity_end + "|" +entity_form + "|" +entity_type + "\n")
                inside = False

        if inside : out.write(sid + "|" + entity_start+"-"+entity_end + "|" + entity_form + "|" + entity_type + "\n")
