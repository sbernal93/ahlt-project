#!/usr/bin/env python3

import sys
from sklearn.model_selection import KFold

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras.layers.merge import add
from keras_contrib.layers import CRF
from keras import backend as K

import tensorflow as tf
import tensorflow_hub as hub

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


def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),
                    "sequence_len": tf.constant(batch_size*[max_len])},
                      signature="tokens",
                      as_dict=True)["elmo"]

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

    #t = Tokenizer(oov_token='<unw>')
    #t.fit_on_texts(docs)
    #vocab_size = len(t.word_index) + 1
    #encoded_docs = t.texts_to_sequences(docs)
    #X = pad_sequences(maxlen=max_len, sequences=encoded_docs, padding="post", value=vocab_size-1)

    X = []
    for seq in docs:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("PADword")
        X.append(new_seq)

    X = np.array(X)
    Y = [[tag2idx[t] for t in tag ]for tag in tags]
    Y = pad_sequences(maxlen=max_len, sequences=Y, padding="post", value=tag2idx["O"])
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
    #Y = [to_categorical(i, num_classes=n_tags) for i in Y]

    sess = tf.Session()
    K.set_session(sess)
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())


    batch_size = 32
    #input = Input(shape=(max_len,), dtype=tf.string)
    #model = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input)
    #model = Embedding(input_dim=vocab_size, output_dim=100,
    #                  input_length=max_len, mask_zero=True, weights=[embedding_matrix],
    #                   trainable=False)(input)  # 20-dim embedding
    #model = Embedding(input_dim=n_words + 1, output_dim=450,
    #                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding

    #model = Bidirectional(LSTM(units=512, return_sequences=True,
    #                           recurrent_dropout=0.3))(model)  # variational biLSTM
    #model = TimeDistributed(Dense(100, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    #crf = CRF(n_tags, activation="linear")  # CRF layer
    #out = crf(model)  # output

    input_text = Input(shape=(max_len,), dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    x = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(embedding)
    x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

    model = Model(input_text, out)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X, Y, batch_size=batch_size, epochs=5, validation_split=0.1, verbose=1)
    #kf = KFold(n_splits=5, shuffle=True)
    #Y = np.array(Y)
    #for train_index, test_index in kf.split(X, Y):
        #train_x, val_x = X[train_index], X[test_index]
        #train_y, val_y = Y[train_index], Y[test_index]
        #model.fit(train_x, train_y, batch_size=batch_size, epochs=5, validation_data=(val_x, val_y), verbose=1)

    out = open(sys.argv[3], "w+")
    for xseq,toks in instances_pred(open(sys.argv[2])):
        docs = [[t[0] for t in xseq]]
        encoded_docs = t.texts_to_sequences(docs)
        x_test_sent = pad_sequences(maxlen=max_len, sequences=encoded_docs, padding="post", value=vocab_size-1)

        prediction = model.predict(x_test_sent)
        prediction = np.argmax(prediction[0], axis=-1)
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
