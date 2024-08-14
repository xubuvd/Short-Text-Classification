# -*- coding:utf-8 -*-
import os
import json
import codecs
import pickle
import yaml
import numpy as np
import collections
import networkx as nx
from tqdm import tqdm

def DataReader(input_file,delim_space=False):
    with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
    x_text = []
    y_label = []
    for line in data:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:continue
        groundTruth = int(tokens[0])
        y_label.append(groundTruth)
        if delim_space:
            x_text.extend([tokens[1].strip().split(' ')])
        else:
            x_text.extend([list(tokens[1].strip())])
    assert len(x_text) == len(y_label)
    return x_text,y_label

def load(input_file):
    with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
    return data

if __name__ == "__main__":

    dataset_name = ['CR','CVQD','DanMu','MPQA','MR','SST1','SST2','SUBJ','TREC']
    data_name = ["train","test"]
    
    for dataset in dataset_name:
        corpus_file = "HyperGAT/data/" + dataset + "_corpus.txt"
        labels_file = "HyperGAT/data/" + dataset + "_labels.txt"
        if os.path.exists(corpus_file):os.remove(corpus_file)
        if os.path.exists(labels_file):os.remove(labels_file)

        fcorpus = open(corpus_file, 'w+')
        flabels = open(labels_file, 'w+')

        for data in data_name:
            input_file = "data/" + dataset + "/data/" + data + ".txt"
            with codecs.open(input_file,'rb','utf-8') as f: allines = f.readlines()
            for idx,line in enumerate(allines):
                tokens = line.strip().split('\t')
                label,text = tokens[0],tokens[1]
                fcorpus.write("{}\n".format(text))
                flabels.write("{}\t{}\t{}\n".format(idx,data,label))
        fcorpus.close()
        flabels.close()

