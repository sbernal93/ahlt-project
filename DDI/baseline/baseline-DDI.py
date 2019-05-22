#! /usr/bin/python3

from nltk import pos_tag

import sys
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

advise_clues = ["should", "may", "could", "would"]
effect_clues = ["administered", "concurrently", "concomitantly", "increase", "increases", "increased", "effect",
                "effects", "prevent", "prevents", "prevented", "potentiate", "potentiates", "potentiated"]
mechanism_clues = ["inhibit", "reduce", "reduces", "reduced", "decrease", "decreases", "decreased", "change",
                    "changes", "changed","elevate", "elevates", "elevated", "interfere", "interferes", "interfered"]
int_clues = ["interaction"]

## -------------------
## -- check if a pair has an interaction, of which type
def check_interaction(tokens,entities,e1,e2) :
   ## Complete this function to return a pair (boolean, ddi_type)
   ## depending on whether there is an interaction between e1 and e2
    pos_tags = pos_tag([t[0] for t in tokens])
    lemmas = [wnl.lemmatize(t[0]) for t in tokens]
    before = []
    inBetween = []
    after = []
    isInBetween = False
    isBefore = True
    isAfter = False
    for k in range(len(tokens)):
        t = [tokens[k][0], tokens[k][1], tokens[k][2], pos_tags[k][1], lemmas[k]]
        if str(t[1]) == entities[e1][0]:
            isBefore = False
            isInBetween = True
        elif str(t[1]) == entities[e2][0]:
            isInBetween = False
            isAfter = True
        elif isBefore:
            before.append(t)
        elif isInBetween:
            inBetween.append(t)
        elif isAfter:
            after.append(t)
    if any(e in effect_clues for e in [t[4] for t in inBetween]): return True,"effect"
    if any(e in mechanism_clues for e in [t[4] for t in inBetween]): return True,"mechanism"
    if any(e in advise_clues for e in [t[4] for t in inBetween]): return True,"advise"
    if any(e in int_clues for e in [t[4] for t in before]): return True,"int"

    if any(e in effect_clues for e in [t[4] for t in after]): return True,"effect"
    if any(e in mechanism_clues for e in [t[4] for t in after]): return True,"mechanism"
    if any(e in advise_clues for e in [t[4] for t in after]): return True,"advise"
    if any(e in int_clues for e in [t[4] for t in after]): return True,"int"

    if any(e in effect_clues for e in [t[4] for t in before]): return True,"effect"
    if any(e in mechanism_clues for e in [t[4] for t in before]): return True,"mechanism"
    if any(e in advise_clues for e in [t[4] for t in before]): return True,"advise"
    if any(e in int_clues for e in [t[4] for t in before]): return True,"int"

    return False,"null"


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
           (is_ddi,ddi_type) = check_interaction(tokens,entities,id_e1,id_e2)
           ddi = "1" if is_ddi else "0"
           print(sid+"|"+id_e1+"|"+id_e2+"|"+ddi+"|"+ddi_type)
