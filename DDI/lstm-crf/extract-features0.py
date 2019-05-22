#! /usr/bin/python3

from nltk import pos_tag

import sys
import string
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


advise_pos = ["MD"]
interest_pos = ["NN", "VBZ", "VBN", "VBD", "VB", "MD", "RB", "VBP"]
advise_clues = ["should", "may", "could", "would"]
effect_clues = ["administered", "concurrently", "concomitantly", "increase", "increases", "increased", "effect",
                "effects", "prevent", "prevents", "prevented", "potentiate", "potentiates", "potentiated"]
mechanism_clues = ["inhibit", "reduce", "reduces", "reduced", "decrease", "decreases", "decreased", "change",
                    "changes", "changed","elevate", "elevates", "elevated", "interfere", "interferes", "interfered"]
int_clues = ["interaction"]


## --------- Feature extractor -----------
## -- Extract features for each token in given sentence

def extract_features(tokens,entities,e1,e2) :

    # for each token, generate list of features and add it to the result
    # can add more features, maybe nltk lemmatize and stuff
    result = []
    tokenFeatures = [];

    pos_tags = pos_tag([t[0] for t in tokens])
    lemmas = [wnl.lemmatize(t[0]) for t in tokens]
    before = []
    inBetween = []
    after = []
    isInBetween = False
    isBefore = True
    isAfter = False
    for k in range(len(tokens)):
        #t= token, span start, span end, pos, lemma
       t = [tokens[k][0].lower(), tokens[k][1], tokens[k][2], pos_tags[k][1], lemmas[k], pos_tag(lemmas[k])[0][1]]
       if str(t[1]) == entities[e1][0]:
           isBefore = False
           isInBetween = True
           tokenFeatures.append("pos1="+pos_tags[k][1])
           tokenFeatures.append("lemma1="+lemmas[k])
           if(k!= len(tokens)-1):
               tokenFeatures.append("nextToken2="+tokens[k+1][0])
           if(k>0):
               tokenFeatures.append("prevToken1="+tokens[k-1][0])
       elif str(t[1]) == entities[e2][0]:
            isInBetween = False
            isAfter = True
            tokenFeatures.append("pos2="+pos_tags[k][1])
            tokenFeatures.append("lemma2="+lemmas[k])
            if(k!= len(tokens)-1):
                tokenFeatures.append("nextToken2="+tokens[k+1][0])
            if(k>0):
               tokenFeatures.append("prevToken1="+tokens[k-1][0])
       elif isBefore:
           before.append(t)
       elif isInBetween:
           inBetween.append(t)
       elif isAfter:
           after.append(t)
    #if any(e in effect_clues for e in [t[4] for t in inBetween]): tokenFeatures.append("isEffectInBetween")
    #if any(e in mechanism_clues for e in [t[4] for t in inBetween]): tokenFeatures.append("isMechanismInBetween")
    #if any(e in advise_clues for e in [t[4] for t in inBetween]): tokenFeatures.append("isAdviseInBetween")
    #if any(e in int_clues for e in [t[4] for t in inBetween]): tokenFeatures.append("isIntInBetween")

    #if any(e in effect_clues for e in [t[4] for t in after]): tokenFeatures.append("isEffectInAfter")
    #if any(e in mechanism_clues for e in [t[4] for t in after]): tokenFeatures.append("isMechanismAfter")
    #if any(e in advise_clues for e in [t[4] for t in after]): tokenFeatures.append("isAdviseInBetween")
    #if any(e in int_clues for e in [t[4] for t in after]): tokenFeatures.append("isIntAfter")

    #if any(e in effect_clues for e in [t[4] for t in before]): tokenFeatures.append("isEffectBefore")
    #if any(e in mechanism_clues for e in [t[4] for t in before]): tokenFeatures.append("isMechanismBefore")
    #if any(e in advise_clues for e in [t[4] for t in before]): tokenFeatures.append("isAdviseBefore")
    #if any(e in int_clues for e in [t[4] for t in before]): tokenFeatures.append("isIntAfter")


    searchList = [['inBetween', inBetween], ['before', before], ['after', after]]
    for ele in searchList:
        c = 0
        i = 0
        for t in ele[1]:
            if(t[4] in stop_words):
                tokenFeatures.append(str(c) + "StopWord=" + t[4])
            if(t[4] in string.punctuation):
                tokenFeatures.append(str(c) + "Punct=" + t[4])
            #tokenFeatures.append(ele[0] + str(c) + "Word="+t[0])
            #tokenFeatures.append(ele[0] + str(c) + "Lemma="+t[4])
            #tokenFeatures.append(ele[0] + str(c) + "Pos="+t[3])
            #tokenFeatures.append(ele[0] + str(c) + "LemmaPos="+t[5])
            #if t[4] in effect_clues:
            #    tokenFeatures.append(ele[0] + "Effect" + str(c) + "Lemma=" + t[4])
            #    tokenFeatures.append(ele[0] + "Effect" + str(c) + "LemmaPos=" + t[4])
            #    tokenFeatures.append(ele[0] + "Effect" + str(c) + "Pos=" + t[3])
            #    tokenFeatures.append(ele[0] + "Effect" + str(c) + "=" + t[0])
            #if t[4] in mechanism_clues:
            #    tokenFeatures.append(ele[0] + "Mech" + str(c) + "Lemma=" + t[4])
            #    tokenFeatures.append(ele[0] + "Mech" + str(c) + "LemmaPos=" + t[4])
            #    tokenFeatures.append(ele[0] + "Mech" + str(c) + "Pos=" + t[3])
            #    tokenFeatures.append(ele[0] + "Mech" + str(c) + "=" + t[0])
            #if t[4] in advise_clues:
            #    tokenFeatures.append(ele[0] + "Advise" + str(c) + "Lemma=" + t[4])
            #    tokenFeatures.append(ele[0] + "Advise" + str(c) + "LemmaPos=" + t[4])
            #    tokenFeatures.append(ele[0] + "Advise" + str(c) + "Pos=" + t[3])
            #    tokenFeatures.append(ele[0] + "Advise" + str(c) + "=" + t[0])
            #if t[4] in int_clues:
            #    tokenFeatures.append(ele[0] + "Int" + str(c) + "Lemma=" + t[4])
            #    tokenFeatures.append(ele[0] + "Int" + str(c) + "LemmaPos=" + t[4])
            #    tokenFeatures.append(ele[0] + "Int" + str(c) + "Pos=" + t[3])
            #    tokenFeatures.append(ele[0] + "Int" + str(c) + "=" + t[0])
            if t[4].isnumeric():
                tokenFeatures.append(ele[0] + str(c) + "Number=" + t[4])
            if t[4] in effect_clues or t[4] in mechanism_clues or t[4] in advise_clues or t[4] in int_clues:
                tokenFeatures.append(ele[0] + str(c) + "Lemma=" + t[4])
                tokenFeatures.append(ele[0] + str(c) + "LemmaPos=" + t[4])
                tokenFeatures.append(ele[0] + str(c) + "Pos=" + t[3])
                tokenFeatures.append(ele[0] + str(c) + "=" + t[0])
                c = c + 1
            if t[3] in interest_pos:
                tokenFeatures.append(ele[0] + str(i) + "Interest="+t[0])
                tokenFeatures.append(ele[0] + str(i) + "InterestLemma=" + t[4])
                tokenFeatures.append(ele[0] + str(i) + "InterestLemmaPos=" + t[4])
                tokenFeatures.append(ele[0] + str(i) + "InterestPos=" + t[3])
                i = i + 1

    return tokenFeatures


## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    for t in word_tokenize(txt):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
    return tks

## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

stop_words = set(stopwords.words('english'))
# directory with files to process
datadir = sys.argv[1]
wnl = WordNetLemmatizer()
for f in listdir(datadir) :

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text

        tokens = tokenize(stext)

        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")
           entities[id] = offs

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
           id_e1 = p.attributes["e1"].value
           id_e2 = p.attributes["e2"].value
           features = extract_features(tokens,entities,id_e1,id_e2)
           ddi = p.attributes["ddi"].value
           type = "null"
           if ddi == "true":
               try:
                   type = p.attributes["type"].value
               except:
                   pass
           #print(sid+"|"+id_e1+"|"+id_e2+"|"+ddi+"|"+ddi_type)
           print (sid, id_e1, id_e2, ddi, type, "\t".join(features), sep='\t')
           print()
