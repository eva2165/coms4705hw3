from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from extract_training_data import FeatureExtractor, State

def is_legal(action, state):
    t, l = action
    empty_stack = len(state.stack) == 0
    if (t == 'left_arc' or t == 'right_arc') and empty_stack:
        return False
    elif (t == 'shift') and (len(state.buffer) == 1) and not empty_stack:
        return False
    elif (t == 'left_arc') and state.stack[-1] == 0:
        return False
    else:
        return True

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            actions = self.model.predict(\
                    self.extractor.get_input_representation(\
                                words, pos, state).reshape(1,6)).tolist()[0]
            s_actions = sorted(enumerate(actions), key=lambda x: x[1], reverse=True)
            
            i, found = 0, False
            while i < len(s_actions) and not found:
                label = self.output_labels[s_actions[i][0]]
                if is_legal(label, state):
                    found = True
                    if label[0] == 'shift':
                        state.shift()
                    elif label[0] == 'left_arc':
                        state.left_arc(label[1])
                    elif label[0] == 'right_arc':
                        state.right_arc(label[1])
                    else:
                        raise ValueError(f"unrecognized label {label[1]}")
                i += 1
            if not found:
                raise RuntimeError("found no valid state transition")
        
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
