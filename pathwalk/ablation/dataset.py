# -*- coding:utf-8 -*-
import os
import codecs
import pickle
import torch
import random
import numpy as np
import collections
import networkx as nx
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset
from vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence

class GraphWalk(object):
    def __init__(self):
        pass
    @classmethod
    def in_degree_edges(cls,G,nodes,edge2id):
        graph_path = []
        in_edges = []
        original_nodeset = set(nodes)
        for j in range(len(nodes)):
            node_id = nodes[j]
            #nbrs_in = set(G.predecessors(node_id)) - original_nodeset
            graph_path.append(node_id)
            #links = []
            #for in_node in nbrs_in:
            #    edge = (in_node,node_id)
            #    links.append(edge2id.get(edge))
            #outer_edges.append(links)
            if j > 0 and (nodes[j-1],node_id) in edge2id:
                edge_id = edge2id.get((nodes[j-1],node_id))
                in_edges.append(edge_id)
        return in_edges

    @classmethod
    def random_sampling_edges(cls,G,nodes,edge2id,select_indegree_num):
        inedges = list()
        inner_edges = []
        outer_edges = []
        for j in range(len(nodes)):
            node_id = nodes[j]
            outer_edges.extend(np.random.choice(list(edge2id.values()),select_indegree_num))
            #inner-links wthin S
            if j > 0 and (nodes[j-1],node_id) in edge2id:
                edge = (nodes[j-1],node_id)
                inner_edges.append(edge2id.get(edge))
        inedges.extend(inner_edges)
        inedges.extend(outer_edges)
        return inedges

    """
    seek to find higher order simplical complex, e.g., 2-complex, 3-complex in graph
    """
    @classmethod
    def find_loop_simplices(cls,G,X,edge2id):
        in_out_list = list()
        for k in range(len(X)):
            current_node = X[k]
            nbrs_out = set(nx.neighbors(G, current_node)) - set(X)
            nbrs_in = set(G.predecessors(current_node)) - set(X)
            if k == 0: in_out_list.append([set(),nbrs_out])
            elif k == len(X) - 1: in_out_list.append([nbrs_in,set()])
            else: in_out_list.append([nbrs_in,nbrs_out])
        prefix_node = [list() for i in range(len(X))]
        outer_node = [list() for i in range(len(X))]
        successor_node = [list() for i in range(len(X))]
        edge1 = [list() for i in range(len(X))]
        edge2 = [list() for i in range(len(X))]
        #stat_dict = dict()
        for i in range(1,len(in_out_list)):
            for j in range(0,i+1):# i+1 which means self-loop outer nodes
                node_share = in_out_list[j][1] & in_out_list[i][0]
                #t2_list = [{X[j],X[i],node} for node in node_share]
                for out_node in node_share:
                    out_edge1 = (X[j],out_node); out_edge2 = (out_node,X[i])
                    edge1_id = edge2id.get(out_edge1); edge2_id = edge2id.get(out_edge2)
                    outer_node[i].append(out_node)
                    prefix_node[i].append(X[j])
                    successor_node[i].append(X[i])
                    edge1[i].append(edge1_id);
                    edge2[i].append(edge2_id)
                    #if i-j not in stat_dict: stat_dict[i-j] = 0
                    #stat_dict[i-j] += 1
        # randomly sample meta-path based neighbors
        for i in range(len(outer_node)):
            if len(outer_node[i]) < 1: continue
            lzip = list(zip(outer_node[i],prefix_node[i],successor_node[i],edge1[i],edge2[i]))
            random.shuffle(lzip)
            outer_node[i][:],prefix_node[i][:],successor_node[i][:],edge1[i][:],edge2[i][:] = zip(*lzip)
        #for key in stat_dict:
        #    print("Implementation\t{}\t{}".format(key,stat_dict[key]))
        return prefix_node,edge1,outer_node,edge2,successor_node

    @classmethod
    def random_sampling_simplices(cls,X):
        simplex_list = list()
        if self.param_dict["select_indegree_num"] <= 0:return None
        (m,n) = np.shape(X)
        for i in range(m):
            in_out_list = list()
            x_tensor = X[i]
            node_simplices = [list() for i in range(len(x_tensor))]
            for i in range(len(x_tensor)):
                current_node = x_tensor[i]
                if int(current_node) < 1:continue
                sample_nodes = np.random.choice(list(self.G.nodes),self.param_dict["select_indegree_num"]) 
                node_simplices[i].extend(sample_nodes)
            simplex_list.append(node_simplices)
        return simplex_list

def DataReader(input_file,delim_space=False,overfit=False,tokenizer=None):
    with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
    x_text = []
    y_label = []
    for line in data:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:continue
        groundTruth = int(tokens[0])
        y_label.append(groundTruth)
        if tokenizer is None:
            if delim_space:
                x_text.extend([tokens[1].strip().split(' ')])
            else:
                x_text.extend([list(tokens[1].strip())])
        else:
            token = ['[CLS]'] + tokenizer.tokenize(tokens[1].strip())
            token_ids = tokenizer.convert_tokens_to_ids(token)
            x_text.extend([token_ids])
        if overfit and len(x_text) > 640: break
    assert len(x_text) == len(y_label)
    return x_text,y_label

def pad_sentence(sequence: List[int],max_sequence_length=10,vocabulary=None):
    sequence_length = len(sequence)
    # Pad sequence to max_sequence_length.
    maxpadded_sequence = torch.full(
        (1,max_sequence_length),
        fill_value=vocabulary.PAD_INDEX
    )
    node_mask = torch.full(
        (1,max_sequence_length),
        fill_value=0
    )
    node_mask[:sequence_length] = 1

    padded_sequence = pad_sequence(
        [torch.tensor(sequence)],
        batch_first=True,
        padding_value=vocabulary.PAD_INDEX
    )
    maxpadded_sequence[:, : padded_sequence.size(1)] = padded_sequence
    return maxpadded_sequence, sequence_length,node_mask

def pad_matrix(sequences:List[List[int]], max_sequence_length=10, select_indegree_num=5, vocabulary=None):
    
    node_mask = torch.full(
        (max_sequence_length,select_indegree_num),
        fill_value=0
    )
    for i in range(len(sequences)):
        sequences[i] = sequences[i][:select_indegree_num]
        node_mask[i][:len(sequences[i])] = 1

    while len(sequences) < max_sequence_length: sequences.append([vocabulary.PAD_INDEX]*select_indegree_num)
    sequence_lengths = [len(sequence) for sequence in sequences]
    maxpadded_sequences = torch.full(
        (len(sequences),select_indegree_num),
        fill_value=vocabulary.PAD_INDEX
    )
    padded_sequences = pad_sequence(
        [torch.tensor(sequence) for sequence in sequences],
        batch_first=True,
        padding_value=vocabulary.PAD_INDEX,
    )
    maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences

    return maxpadded_sequences,sequence_lengths,node_mask

"""
data handle for training
loop_topology: True, sampling loop topologies as features; False, random sampling edges and nodes as additional features
"""
class PathWalkDataset(Dataset):
    def __init__(self,config,input_file,graph,edge2id,use_word=False,loop_topology=True,overfit=False,tokenizer=None):
        super().__init__()
        self.config = config
        self.loop_topology = loop_topology
        self.tokenizer = tokenizer
        self.vocabulary = Vocabulary(
            config["word_counts_json"],
            min_count=config["vocab_min_count"]
        )
        self.vocab_size = len(self.vocabulary.node_ids)
        self.x_text,self.y_label = DataReader(input_file=input_file,delim_space=use_word,overfit=overfit,tokenizer=tokenizer)
        self.G = graph
        self.edge_size = self.G.number_of_edges()
        self.edge2id = edge2id
        print("vocabulary:",len(self.vocabulary))
        print("number_of_nodes:",self.G.number_of_nodes())
        assert len(self.vocabulary) == self.G.number_of_nodes()

    def __len__(self):
        return len(self.x_text)
    def __getitem__(self, index):
        item = {}

        x = self.x_text[index]
        x = self.vocabulary.to_indices(x)
        x = x[: self.config["max_sequence_length"]]
        y_label = self.y_label[index]
        
        # graph path
        in_edges = GraphWalk.in_degree_edges(self.G,x,self.edge2id)
        x_paded,x_len,x_mask = pad_sentence(x,self.config["max_sequence_length"],self.vocabulary)
        in_edges_paded,_,inedge_mask = pad_sentence(in_edges,self.config["max_sequence_length"],self.vocabulary)
        
        # graph attention over neighbourhood nodes
        prefix_node,edge1,outer_node,edge2,successor_node = GraphWalk.find_loop_simplices(self.G,x,self.edge2id)
        prefix_paded,prefix_len,prefix_mask = pad_matrix(
            prefix_node,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        edge1_paded,edge1_len,edge1_mask = pad_matrix(
            edge1,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        outer_paded,outer_len,outer_mask = pad_matrix(
            outer_node,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        edge2_paded,edge2_len,edge2_mask = pad_matrix(
            edge2,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        successor_paded,successor_len,successor_mask = pad_matrix(
            successor_node,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        
        item["x"] = x_paded.long()
        item["y"] = torch.tensor(y_label).long()
        item["xlen"] = torch.tensor(x_len).long()
        item["in_edges"] = in_edges_paded.long()
        item["x_mask"] = x_mask.long()
        item["inedge_mask"] = inedge_mask.long()
        item["prefix_node"] = prefix_paded.long()
        item["prefix_mask"] = prefix_mask.long()
        item["edge1"] = edge1_paded.long()
        item["outer_node"] = outer_paded.long()
        item["outer_mask"] = outer_mask.long()
        item["edge2"] = edge2_paded.long()
        item["successor_node"] = successor_paded.long()
        item["successor_mask"] = successor_mask.long()
        return item

"""
Handle for extracting topological structure in training textual datasets
"""
class TopologyData():
    def __init__(self,input_file="",sequence_len=10,batch_size=128,cross_valid=0.2,indegree=5,delim_space=False,loop_topology=True):
        super().__init__()	
        self.param_dict = {"sequence_length":sequence_len,
        "batch_size":batch_size,
        "validation":cross_valid,
        "select_indegree_num":indegree,
        "delim_space":delim_space,
        "loop_topology":loop_topology}
        self.loop_topology = loop_topology
        self.batch_size = batch_size
        self.load_train_data(input_file)
        self.reset_batch()
        self.build_network()

    def load_train_data(self,data_file):
        self.x_text = list()
        self.y_cate = list()
        self.vocab = list()
        self.x_tensor = list()
        with codecs.open(data_file,'rb','utf-8') as f:data = f.readlines()
        for line in data:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:continue
            label = int(tokens[0])
            if self.param_dict["delim_space"]:
                self.vocab.extend(tokens[1].strip().split(' '))
                sent = [item.strip() for item in tokens[1].strip().split(' ') if len(item.strip()) > 0]
            else:
                self.vocab.extend(list(tokens[1]))
                sent = [item.strip() for item in list(tokens[1].strip()) if len(item.strip()) > 0]
            self.y_cate.append(label)
            self.x_text.append(sent)
        self.y_cate = np.array(self.y_cate)
        self.vocab = list(set(self.vocab))
        self.word2id = {w:i+1 for i,w in enumerate(self.vocab)}
        self.node_ids = [i+1 for i,w in enumerate(self.vocab)]
        for sent in self.x_text:self.x_tensor.append(list(map(self.word2id.get, sent)))	
        self.edges = list()
        self.edge2id = dict()
        edge_id = len(self.vocab) + 1
        for sent in self.x_tensor:
            for i in range(len(sent)-1):
                link = (sent[i],sent[i+1])
                if link not in self.edges:
                    self.edges.append(link)
                    self.edge2id[link] = edge_id
                    edge_id += 1
        self.x_tensor = pad_sequences(self.x_tensor, maxlen=self.param_dict["sequence_length"],dtype='int32', padding='post', truncating='post', value=0)
        self.indices = np.arange(0,len(self.x_tensor))
        self.num_batch = int(1.0*len(self.x_tensor)/self.param_dict["batch_size"])
        self.vocab_size = len(self.vocab) + 1

    def next_batch(self):
        x = self.x_tensor[self.indices[self.idx_train:self.idx_train+self.param_dict["batch_size"]]]
        y =   self.y_cate[self.indices[self.idx_train:self.idx_train+self.param_dict["batch_size"]]]
        if self.loop_topology:
            simplices = SuperData.find_loop_simplices(self,x)#self.simplex(x)
            links = SuperData.in_degree_edges(self,x)
        else:
            links = SuperData.random_sampling_edges(self,x)
            simplices = SuperData.random_sampling_simplices(self,x) 
        self.idx_train += self.param_dict["batch_size"]
        return x,y,links,simplices

