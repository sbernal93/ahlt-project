#! /usr/bin/python3

import sys
import string
from os import listdir

import nltk
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

## -------- classify_token ----------
## -- check if a token is a drug, and of which type
suffixes = ["azole", "idine", "amine", "mycin", "xacin", "ostol", "adiol"]
suffixes_drug = ["ine", "cin", "ium"]
suffixes_brand = ["gen"]
suffixes_group = ["ines", "ides", "cins", "oles"]
def classify_token(txt):
   if txt.isupper() or txt[-3:] in suffixes_brand : return True,"brand"
   elif txt[-5:] in suffixes or txt[-3:] in suffixes_drug : return True,"drug"
   elif txt[-4:] in suffixes_group : return True,"group"
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
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
        count += 1

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag -----------
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"

## --------- Feature extractor -----------
## -- Extract features for each token in given sentence

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   # can add more features, maybe nltk lemmatize and stuff
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0]

      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])

      tokenFeatures.append("posTag="+pos_tag([t])[0][1])
      (is_drug, tk_type) = classify_token(t)

      tokenFeatures.append("isClassified") if (is_drug) else tokenFeatures.append("notClassified")
      tokenFeatures.append("tkType="+tk_type) if (tk_type) else tokenFeatures.append("tkType=None")

      tokenFeatures.append("isStopWord") if (t in stop_words)  else tokenFeatures.append("notStopWord")
      tokenFeatures.append("isUpper") if (t.isupper()) else tokenFeatures.append("notUpper")
      tokenFeatures.append("isTitle") if (t.istitle()) else tokenFeatures.append("notTitle")
      tokenFeatures.append("isDigit") if (t.isdigit()) else tokenFeatures.append("notDigit")

      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         tokenFeatures.append("isUpperPrev") if (t.isupper()) else tokenFeatures.append("notUpperPrev")
         tokenFeatures.append("isTitlePrev") if (t.istitle()) else tokenFeatures.append("notTitlePrev")
         tokenFeatures.append("isDigitPrev") if (t.isdigit()) else tokenFeatures.append("notDigitPrev")
      else :
         #tokenFeatures.append("BoS")
         tokenFeatures.append("formPrev=")
         tokenFeatures.append("formlowerPrev=")
         tokenFeatures.append("suf3Prev=")
         tokenFeatures.append("suf4Prev=")
         tokenFeatures.append("notUpperPrev")
         tokenFeatures.append("notTitlePrev")
         tokenFeatures.append("notDigitPrev")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         tokenFeatures.append("isUpperNext") if (t.isupper()) else tokenFeatures.append("notUpperNext")
         tokenFeatures.append("isTitleNext") if (t.istitle()) else tokenFeatures.append("notTitleNext")
         tokenFeatures.append("isDigitNext") if (t.isdigit()) else tokenFeatures.append("notDigitNext")
      else:
         #tokenFeatures.append("EoS")
         tokenFeatures.append("formNext=")
         tokenFeatures.append("formlowerNext=")
         tokenFeatures.append("suf3Next=")
         tokenFeatures.append("suf4Next=")
         tokenFeatures.append("notUpperNext")
         tokenFeatures.append("notTitleNext")
         tokenFeatures.append("notDigitNext")

      result.append(tokenFeatures)

   return result


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :

   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)

   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))


      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans)
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
