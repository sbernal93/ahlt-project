#! /usr/bin/python3

import sys
import string
import os
from os import listdir
import pandas as pd

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

## -------- classify_token ----------
## -- check if a token is a drug, and of which type

suffixes = ["azole", "idine", "amine", "mycin", "xacin", "ostol", "adiol"]
suffixes_drug = ["ine", "cin", "ium"]
suffixes_brand = ["gen"]
suffixes_group = ["ines", "ides", "cins", "oles"]
def classify_token(txt):
   ## Complete this function to return a pair (boolean, drug_type)
   ## depending on whether the token is a drug name or not
   if txt[-5:] in suffixes or txt[-3:] in suffixes_drug or txt.lower() in drugnames or txt.lower() in synonyms: return True,"drug"
   elif txt.isupper() or txt[-3:] in suffixes_brand : return True,"brand"
   elif txt[-4:] in suffixes_group or "agent" in txt : return True,"group"
   else : return False,""


## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    tokens = word_tokenize(txt)
    count = 0
    for t in tokens:
        if ("agent" in t) or (t in stop_words) or (t in string.punctuation):
            count += 1
            continue
        if((count + 1 < len(tokens)) and ("agent" in tokens[count + 1]) and
         (((len(wn.synsets(t)) > 1) and (wn.synsets(t)[0].pos() == 'n')) or
         (len(wn.synsets(t)) > 1))):
            t = t + " " + tokens[count + 1]
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
        count += 1

    ## tks is a list of triples (word,start,end)
    return tks


## --------- Entity extractor -----------
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def extract_entities(stext) :

 # convert the sentece to a list of tokens
 tokens = tokenize(stext)

 # for each token, check whether it is a drug name or not
 result = []
 for t in tokens:
    tokenTxt = t[0]
    (is_drug, tk_type) = classify_token(tokenTxt)

    if is_drug :
       drug_start = t[1]
       drug_end = t[2]
       drug_type = tk_type
       e = { "offset" : str(drug_start)+"-"+str(drug_end),
             "text" : stext[drug_start:drug_end + 1],
             "type" : drug_type  }
       result.append(e)


 return result

## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --

stop_words = set(stopwords.words('english'))
# directory with files to process
datadir = sys.argv[1]
db = pd.read_csv('../drugbank/drugbank_vocabulary.csv')
drugnames = [d.lower() for d in db['Common name'].values.tolist()]

split = [str(d).lower().split('|') for d in db['Synonyms'].values.tolist()]
synonyms = [item[1:-1] for sublist in split for item in sublist]

# process each file in directory
for f in listdir(datadir) :

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text

        # extract entities in text
        entities = extract_entities(stext)

        # print sentence entities in format requested for evaluation
        for e in entities :
            print(sid+"|"+e["offset"]+"|"+e["text"]+"|"+e["type"])

##extra commands:
##  java -jar ../../eval/evaluateNER.jar ../../data/Test-NER/DrugBank task9.1_UC3M_1.txt
##  cat ../../data/Train/DrugBank/Amiodarone_ddi.xml | xmllint - format - | grep entity | awk '{print $(NF-1), $NF}'
