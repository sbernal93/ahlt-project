#!/usr/bin/env python3

import sys
from sklearn.model_selection import KFold

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras_contrib.layers import CRF
from keras.optimizers import RMSprop
from keras.utils import plot_model

import matplotlib.pyplot as plt

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

def load_glove():
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('../../../data/glove.6B/glove.6B.100d.txt')
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    return embeddings_index

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

    max_len = 76
    words = list(set(x[0].values))
    words.append("ENDPAD")
    n_words = len(words)
    utags = list(set(y))
    n_tags = len(utags)

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(utags)}

    docs = [[w[0] for w in s] for s in sentences]

    pos_tags = list(set(x[4].values))
    pos_tags.append("ENDPAD")
    n_pos = len(pos_tags)
    pos2idx = {p: i + 1 for i, p in enumerate(pos_tags)}

    X_pos = [[pos2idx[w[4]] for w in s] for s in sentences]
    X_pos = pad_sequences(maxlen=max_len, sequences=X_pos, padding="post", value=n_pos-1)

    suff3 = list(set(x[2].values))
    suff3.append("ENDPAD")
    n_suff3 = len(suff3)
    suff32idx = {s: i + 1 for i, s in enumerate(suff3)}

    X_suff3 = [[suff32idx[w[2]] for w in s] for s in sentences]
    X_suff3 = pad_sequences(maxlen=max_len, sequences=X_suff3, padding="post", value=n_suff3-1)

    suff4 = list(set(x[3].values))
    suff4.append("ENDPAD")
    n_suff4 = len(suff4)
    suff42idx = {s: i + 1 for i, s in enumerate(suff4)}

    X_suff4 = [[suff42idx[w[3]] for w in s] for s in sentences]
    X_suff4 = pad_sequences(maxlen=max_len, sequences=X_suff4, padding="post", value=n_suff4-1)

    tktypes = list(set(x[5].values))
    tktypes.append("ENDPAD")
    n_tktypes = len(tktypes)
    tktypes2idx = {s: i + 1 for i, s in enumerate(tktypes)}

    X_tktypes = [[tktypes2idx[w[5]] for w in s] for s in sentences]
    X_tktypes = pad_sequences(maxlen=max_len, sequences=X_tktypes, padding="post", value=n_tktypes-1)
    #docs.append("UNKNOWN")

    #X = [[word2idx[w[0]] for w in s] for s in sentences]
    #X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)

    t = Tokenizer(oov_token='<unw>')
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(docs)
    X = pad_sequences(maxlen=max_len, sequences=encoded_docs, padding="post", value=vocab_size-1)

    embeddings_index = load_glove()
    embedding_matrix = np.zeros((vocab_size, 100))
    #print(t.word_index.items())
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        #else:
        #    embedding_matrix[i] = embeddings_index.get("unknown")

    Y = [[tag2idx[t] for t in tag ]for tag in tags]
    Y = pad_sequences(maxlen=max_len, sequences=Y, padding="post", value=tag2idx["O"])
    Y = [to_categorical(i, num_classes=n_tags) for i in Y]


    #input = Input(shape=(max_len,))
    #model = Embedding(input_dim=vocab_size, output_dim=100,
    #                  input_length=max_len, mask_zero=True, weights=[embedding_matrix],
    #                   trainable=False)(input)  # 20-dim embedding
    #model = Embedding(input_dim=n_words + 1, output_dim=450,
    #                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    #model = Bidirectional(LSTM(units=64, return_sequences=True,
    #                           recurrent_dropout=0.8))(model)  # variational biLSTM
    #model = TimeDistributed(Dense(100, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    #crf = CRF(n_tags, activation="linear")  # CRF layer
    #out = crf(model)  # output

    #model = Model(input, out)
    #model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])


    word_in = Input(shape=(max_len,))
    word_emb = Embedding(input_dim=vocab_size, output_dim=100,
                      input_length=max_len, mask_zero=True, weights=[embedding_matrix],
                       trainable=False)(word_in)  # 20-dim embedding
    #model = Embedding(input_dim=n_words + 1, output_dim=450,
    #                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    pos_in = Input(shape=(max_len,))
    pos_emb = Embedding(input_dim=n_pos, output_dim=100,
                        input_length=max_len, mask_zero=True)(pos_in)

    suff3_in = Input(shape=(max_len,))
    suff3_emb = Embedding(input_dim=n_suff3, output_dim=100,
                        input_length=max_len, mask_zero=True)(suff3_in)

    suff4_in = Input(shape=(max_len,))
    suff4_emb = Embedding(input_dim=n_suff4, output_dim=100,
                        input_length=max_len, mask_zero=True)(suff4_in)

    tktypes_in = Input(shape=(max_len,))
    tktypes_emb = Embedding(input_dim=n_tktypes, output_dim=100,
                        input_length=max_len, mask_zero=True)(tktypes_in)

    concat = concatenate([word_emb, pos_emb,
#        suff3_emb,
#        suff4_emb,
        tktypes_emb])
    concat = Dropout(0.2)(concat)
    model = Bidirectional(LSTM(units=32, return_sequences=True,
                               recurrent_dropout=0.5))(concat)  # variational biLSTM
    model = TimeDistributed(Dense(10, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags, activation='linear')  # CRF layer
    out = crf(model)  # output

    model = Model([word_in, pos_in,
    # suff3_in,
    # suff4_in,
     tktypes_in], out)

    optimizer = RMSprop(lr=0.01, epsilon=None, decay=0.0)
    model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    #plot_model(model, to_file='model.png')
    model.fit([X, X_pos,
     #X_suff3,
     #X_suff4,
     X_tktypes],
        np.array(Y), batch_size=32, epochs=10,
        validation_split=0.1, verbose=1)
    kf = KFold(n_splits=5, shuffle=True)
    Y = np.array(Y)
    for train_index, test_index in kf.split(X, Y):
        train_x, val_x = X[train_index], X[test_index],
        train_x_pos, val_x_pos = X_pos[train_index], X_pos[test_index]
        train_x_suff3, val_x_suff3 = X_suff3[train_index], X_suff3[test_index]
        train_x_suff4, val_x_suff4 = X_suff4[train_index], X_suff4[test_index]
        train_x_tktype, val_x_tktype = X_tktypes[train_index], X_tktypes[test_index]

        train_y, val_y = Y[train_index], Y[test_index]
#        model.fit([train_x, train_x_pos,
#                    train_x_suff3,
#                    train_x_suff4,
#                    train_x_tktype],
#            train_y, batch_size=32, epochs=5,
#            validation_data=([val_x, val_x_pos,
    #         val_x_suff3,
    #         val_x_suff4,
#             val_x_tktype], val_y), verbose=1)
    #history = model.fit(X, npself.array(Y), batch_size=32, epochs=5,
                #validation_split=0.2, verbose=1)

    #hist = pd.DataFrame(history.history)

    out = open(sys.argv[3], "w+")
    for xseq,toks in instances_pred(open(sys.argv[2])):
        #x = pd.DataFrame(xseq)
        #x_test_sent = [[word2idx.get(t[0],0) for t in xseq]]
        docs = [[t[0] for t in xseq]]
        pos = [[pos2idx.get(t[4],0) for t in xseq]]
        suff3 = [[suff32idx.get(t[2],0) for t in xseq]]
        suff4 = [[suff42idx.get(t[3],0) for t in xseq]]
        tktypes = [[tktypes2idx.get(t[5],0) for t in xseq]]
        encoded_docs = t.texts_to_sequences(docs)
        x_test_sent = pad_sequences(maxlen=max_len, sequences=encoded_docs, padding="post", value=vocab_size-1)
        x_test_pos = pad_sequences(maxlen=max_len, sequences=pos, padding="post", value=n_pos-1)
        x_test_suff3 = pad_sequences(maxlen=max_len, sequences=suff3, padding="post", value=n_suff3-1)
        x_test_suff4 = pad_sequences(maxlen=max_len, sequences=suff4, padding="post", value=n_suff4-1)
        x_test_tktypes = pad_sequences(maxlen=max_len, sequences=tktypes, padding="post", value=n_tktypes-1)

        prediction = model.predict([x_test_sent, x_test_pos,
    #    x_test_suff3,
    #    x_test_suff4,
        x_test_tktypes])
        prediction = np.argmax(prediction[0], axis=-1)
        inside = False;
        for k in range(0,len(toks)) :
            #prediction_max = np.argmax(prediction[k], axis=-1)
            #y = utags[prediction_max[k]]
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
