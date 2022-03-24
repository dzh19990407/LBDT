# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re
import random
import codecs

import numpy as np

import torch
from torchtext.vocab import GloVe

UNK_TOKEN = "<unk>"
PAD_TOKEN = ""
SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")

OPTIONS_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx


class Corpus(object):
    def __init__(self, glove_path):
        self.dictionary = Dictionary()
        self.glove = GloVe(name="840B", dim=300, cache=glove_path)

    def set_max_len(self, value):
        self.max_len = value

    def load_file(self, filename):
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                line = line.strip()
                self.add_to_corpus(line)
        self.dictionary.add_word(UNK_TOKEN)
        self.dictionary.add_word(PAD_TOKEN)

    def get_random_phrase(self, total_words=10):
        vocabulary = list(self.dictionary.word2idx.keys())
        words = random.choices(vocabulary, k=total_words)
        phrase = " ".join(words)
        return phrase


    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=20):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())
        # words = [w.lower() for w in words if len(w) > 0]
        words = [w.lower() for w in words if len(w) > 0 and w != ' ']
        if len(words) == 0:
            words = [""]

        phrase_mask = [1] * len(words)

        if words[-1] == ".":
            words = words[:-1]
            phrase_mask = phrase_mask[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
                phrase_mask = phrase_mask[:max_len]
            elif len(words) < max_len:
                words += [PAD_TOKEN] * (max_len - len(words))
                phrase_mask += [0] * (max_len - len(phrase_mask))

        tokens = [self.glove[word] for word in words]
        tokens = torch.stack(tokens)

        phrase_mask = torch.from_numpy(np.array(phrase_mask))
        return tokens, phrase_mask

    def __len__(self):
        return len(self.dictionary)
