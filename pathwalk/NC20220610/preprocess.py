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
from vocabulary import Vocabulary

'''
 18                 label,content = lin.split('\t')
 19                 token = config.tokenizer.tokenize(content)
 20                 token = [CLS] + token
 21                 seq_len = len(token)
 22                 mask = []
 23                 token_ids = config.tokenizer.convert_tokens_to_ids(token)
'''

def test(input_file):
    with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
    distance_featureNums = dict()
    line_nums = 0
    for line in data:
        if line.find("Implementation") == -1: continue
        tokens = line.strip().split('\t')
        if len(tokens) != 3:continue
        line_nums += 1
        distance = int(tokens[1])
        num = int(tokens[2])
        if distance not in distance_featureNums: distance_featureNums[distance] = 0
        distance_featureNums[distance] += num
    for distance in distance_featureNums:
        nums = distance_featureNums[distance]
        print("distance:{}\ttotal:{}\tavg:{}".format(distance,nums,1.0*nums/line_nums))



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

def gen_wordCountJson(input_file=None,output_json_file=None,delim_space=False,min_count=5):
    x_text,y_label = DataReader(input_file,delim_space)
    text = []
    for item in x_text:text += item
    word_count_dict = collections.Counter(text)
    word_filter_dict = {
        word:cnt 
        for word,cnt in word_count_dict.items()
        if cnt >= min_count
    }
    if os.path.exists(output_json_file): os.remove(output_json_file)
    with open(output_json_file, 'w') as fp: json.dump(word_filter_dict,fp,ensure_ascii=False)
    print("write to {} ok.".format(output_json_file))

def gen_DiGraph(input_file=None,output_graph_gpickle=None,output_edge2id_pkl=None,word_counts_file=None,delim_space=False):

    if os.path.exists(output_graph_gpickle): os.remove(output_graph_gpickle)
    if os.path.exists(output_edge2id_pkl): os.remove(output_edge2id_pkl)

    x_text,y_label = DataReader(input_file,delim_space=delim_space)
    vocabulary = Vocabulary(
        word_counts_file,
        min_count=5
    )
    G = nx.DiGraph()
    node_ids = vocabulary.node_ids
    edges = []
    edge_id = len(node_ids)
    edge2id = {}

    for sent in tqdm(x_text,total=len(x_text)):
        for i in range(len(sent) - 1):
            edge = (vocabulary.word2index.get(sent[i],vocabulary.UNK_INDEX),\
                vocabulary.word2index.get(sent[i+1],vocabulary.UNK_INDEX))
            if edge not in edges:
                edges.append(edge)
                edge2id[edge] = edge_id
                edge_id += 1
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)
    nx.write_gpickle(G,output_graph_gpickle)
    print("wirte to {} ok.".format(output_graph_gpickle))
    with open(output_edge2id_pkl, 'wb') as f: pickle.dump(edge2id, f)
    print("write to {} ok.".format(output_edge2id_pkl))

def read_graph(gpickle_file):
    print("Analysis the file of {} ...".format(gpickle_file))
    G = nx.read_gpickle(gpickle_file)
    print("number_of_edges:{}".format(G.number_of_edges()))
    print("number_of_nodes:{}".format(G.number_of_nodes()))
    res = G.in_degree(G.nodes())
    #print("in_degree of all nodes in G: {}".format(res))
    sum_in_degree = 0.0
    for (node, in_degree) in res:
        sum_in_degree += in_degree
    print("avg in-degree: {}".format(1.0*sum_in_degree/G.number_of_nodes()))

def load(input_file):
    with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
    return data

def stat(input_file):
    data = load(input_file)
    lines = 0
    lsum = 0
    len_list = list()
    for line in data:
        tokens = line.strip().split('\t')
        sent = list(tokens[1].strip().split(" "))
        lsum += len(sent)
        lines += 1
        len_list.append(len(sent))
    print("avgWords:{}".format(1.0*lsum/lines))
    lensent = np.array(len_list)
    print("avg:{}\tstd:{}".format(np.mean(lensent),np.std(lensent)))

def gen_train_dataset(train_data_file,delim_space=False):
    print("preprocess {} ...".format(train_data_file))

    curr_path = os.path.dirname(train_data_file)
    output_word_counts_json = os.path.join(curr_path,"word_counts_train.json")
    output_graph_gpickle = os.path.join(curr_path,"graph.gpickle")
    output_edge2id_pkl = os.path.join(curr_path,"edge2id.pkl")

    stat(train_data_file)

    gen_wordCountJson(
        input_file=train_data_file,
        output_json_file=output_word_counts_json,
        delim_space=delim_space,
        min_count=5
    )

    gen_DiGraph(
        input_file=train_data_file,
        output_graph_gpickle=output_graph_gpickle,
        output_edge2id_pkl=output_edge2id_pkl,
        word_counts_file=output_word_counts_json,
        delim_space=delim_space)
    read_graph(gpickle_file=output_graph_gpickle)
    print("========================================")

if __name__ == "__main__":

    #train_data_file="data/VideoQuery/train.txt"
    #gen_train_dataset(train_data_file)
    #train_data_file="data/THUCNews/raw.txt"
    #gen_train_dataset(train_data_file)


    train_data_file="data/CVQD/raw/CVQD.txt"
    gen_train_dataset(train_data_file,delim_space=False)

    train_data_file="data/DanMu/raw/DanMu.data"
    gen_train_dataset(train_data_file,delim_space=False)

    train_data_file="data/MR/raw/MR.txt"
    gen_train_dataset(train_data_file,delim_space=True)

    train_data_file="data/CR/raw/CR.data"
    gen_train_dataset(train_data_file,delim_space=True)

    train_data_file="data/MPQA/raw/MPQA.data"
    gen_train_dataset(train_data_file,delim_space=True)

    train_data_file="data/SST1/raw/SST1.data"
    gen_train_dataset(train_data_file,delim_space=True)

    train_data_file="data/SST2/raw/SST2.data"
    gen_train_dataset(train_data_file,delim_space=True)

    train_data_file="data/SUBJ/raw/SUBJ.data"
    gen_train_dataset(train_data_file,delim_space=True)

    train_data_file="data/TREC/raw/TREC.data"
    gen_train_dataset(train_data_file,delim_space=True)
    
    #read_graph("data/DanMu/data/graph.gpickle")
    #read_graph("data/MR/data/graph.gpickle")
    #read_graph("data/CVQD/data/graph.gpickle")
	#test("neighbors_stat2.txt")


