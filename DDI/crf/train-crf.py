#!/usr/bin/env python3

import pycrfsuite
import sys

# Inherit crfsuite.Trainer to implement message() function, which receives
# progress messages from a training process.
class Trainer(pycrfsuite.Trainer):
    def message(self, s):
        # Simply output the progress messages to STDOUT.
        sys.stdout.write(s)

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
        # fields are:  0=sid, 1=id_e1, 2=id_e2, 3=interaction, 4=tag, 5...N = features
        item = fields[5:]
        xseq.append(item)

        # Append the label to the label sequence.
        yseq.append(fields[4])



if __name__ == '__main__':
    # Create a Trainer object.
    trainer = Trainer()

    maxnull = 500000
    c = 0
    # Read training instances from STDIN, and append them to the trainer.
    for xseq, yseq in instances(sys.stdin):
        if(yseq[0] == "null") : c = c + 1
        if(yseq[0] == "null" and c > maxnull): continue
        trainer.append(xseq, yseq, 0)

    # Use L2-regularized SGD and 1st-order dyad features.
    #trainer.select('l2sgd', 'crf1d')
    trainer.select('arow', 'crf1d')

    # This demonstrates how to list parameters and obtain their values.
    for name in trainer.params():
        print (name, trainer.get(name), trainer.help(name))

    # Set the coefficient for L2 regularization to 0.1 -> can experiment with this,
    # ignores some features according to relevance
    trainer.set('feature.minfreq', 0.00001)
    #trainer.set('c2', 1)
    trainer.set('gamma', 7)
    trainer.set('variance', 1)
    trainer.set('max_iterations', 90000)

    # Start training; the training process will invoke trainer.message()
    # to report the progress.
    trainer.train(sys.argv[1], -1)
