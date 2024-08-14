""" Created by minhnq """
from __future__ import print_function

import codecs
import collections
import glob
import operator
import os
import re

import numpy as np

_WORD_SPLIT = re.compile("([.,!?\"/':;)(])")
_DIGIT_RE = re.compile(br"\d")
STOP_WORDS = "\" \' [ ] . , ! : ; ?".split(" ")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    return [w.lower() for w in words if w != '' and w != ' ']


def load_vocab(path):
    vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_train_data(train_path):
    classes = len(glob.glob(train_path+"/*.npy"))
    print("num classes = {}".format(classes))
    class_weights = np.zeros(classes)
    result = []
    for cls in range(classes):
        result.append(np.load(os.path.join(train_path,"{}.npy".format(cls))))
        class_weights[cls] = len(result[cls])
    total_tokens = np.sum(class_weights)
    class_weights = total_tokens / class_weights
    class_weights = class_weights / np.mean(class_weights)
    return result, class_weights


def _select_examples(X, maxlen):
    begin = np.random.randint(len(X) - maxlen)
    return X[begin: begin + maxlen]
    pass


def next_batch(X, batch_size, maxlen):
    x = np.zeros(shape=[batch_size, maxlen + 1])
    choices = np.random.randint(len(X), size=batch_size)
    for idx, choice in enumerate(choices):
        x[idx, :-1] = _select_examples(X[choice], maxlen)
        x[idx, -1] = choice
    Y = np.expand_dims(x[:, -1], 1).copy()
    x = x[:, :-1].copy()
    return x, Y

def generate_vntc_data(data_path, save_vocab_path):
    listdirs = [os.path.join(data_path,dir) for dir in os.listdir(data_path)]
    n_classes = len(listdirs)
    # BUILD VOCAB
    vocab = {}
    for file in glob.glob(data_path + "/*/*.txt"):
        with open(file, "rt") as f:
            vocab = build_vocab(vocab, f.read())
    vocab = collections.OrderedDict(sorted(vocab.items(), key = operator.itemgetter(1), reverse = True))
    with open(save_vocab_path, "w") as f:
        for i,v in enumerate(vocab):
            f.write("{} {}\n".format(v,i))
    return vocab

def build_vocab(vocab, text):
    for w in text.lower().split():
        if w in vocab:
            vocab[w] += 1
        else:
            vocab[w] = 1
    return vocab


if __name__ == '__main__':
    # with open("/home/minhnq/DeepLearning/text_classifier/Train_Full/Doi song/DS_TN_ (4794).txt", "rt") as f:
    #     print(f.read())
    generate_vntc_data("./Train_Full","./corpora/vocab.txt")