# -*- coding:utf-8 -*-
import os
import codecs
import pickle
import torch
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
        outer_edges = []
        original_nodeset = set(nodes)
        for j in range(len(nodes)):
            node_id = nodes[j]
            nbrs_out = set(nx.neighbors(G, node_id)) - original_nodeset
            nbrs_all = set(nx.all_neighbors(G, node_id)) - original_nodeset
            nbrs_in  =  nbrs_all - nbrs_out
            graph_path.append(node_id)
            links = []
            for enode in nbrs_in:
                edge = (enode,node_id)
                links.append(edge2id.get(edge))
            outer_edges.append(links)
            #inner-links wthin S
            if j > 0 and (nodes[j-1],node_id) in edge2id:
                edge_id = edge2id.get((nodes[j-1],node_id))
                graph_path.append(edge_id)
        return graph_path,outer_edges

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
    def find_loop_simplices(cls,X):
        (m,n) = np.shape(X)
        simplex_list = list()
        if self.param_dict["select_indegree_num"] <= 0:return None
        for i in range(m):
            in_out_list = list()
            x_tensor = X[i]
            #x_tensor = x_tensor[0]
            for k in range(len(x_tensor)):
                current_node = x_tensor[k]
                if int(current_node) < 1:continue
                nbrs_all = set(nx.all_neighbors(self.G, current_node)) - set(x_tensor)
                nbrs_out = set(nx.neighbors(self.G, current_node)) - set(x_tensor)
                nbrs_in  =  nbrs_all - nbrs_out
                if k == 0:
                    temp = [set(),nbrs_out]
                    in_out_list.append(temp)
                elif k == len(x_tensor) - 1:
                    temp = [nbrs_in,set()]
                    in_out_list.append(temp)
                else:
                    temp = [nbrs_in,nbrs_out]
                    in_out_list.append(temp)
            t2_complex = list()
            t3_complex = list()
            t4_complex = list()
            node_simplices = [list() for i in range(len(x_tensor))]
            for i in range(len(in_out_list)):
                #2-simplex
                if i + 1 < len(in_out_list):
                    node_share = in_out_list[i][1] & in_out_list[i+1][0]
                    t2_list = [{x_tensor[i],node,x_tensor[i+1]} for node in node_share]
                    t2_complex.extend(t2_list)
                    node_simplices[i].extend([node for node in node_share])
                #3-simplex
                if i + 2 < len(in_out_list):
                    node_share = in_out_list[i][1] & in_out_list[i+2][0]
                    t3_list = [{x_tensor[i],x_tensor[i+1],x_tensor[i+2],node} for node in node_share]
                    t3_complex.extend(t3_list)
                    node_simplices[i+2].extend([node for node in node_share])
                #4-simplex
            simplex_list.append(node_simplices)
        return simplex_list

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

def DataReader(input_file,delim_space=False,overfit=False):
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
        if overfit and len(x_text) > 640: break
    assert len(x_text) == len(y_label)
    return x_text,y_label

def pad_sentence(sequence: List[int],max_sequence_length=20,vocabulary=None):
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

def pad_matrix(sequences: List[List[int]],max_sequence_length=10,select_indegree_num=5,vocabulary=None):
    
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
class BatchData(Dataset):
    def __init__(self,config,input_file,graph,edge2id,use_word=False,loop_topology=True,overfit=False):
        super().__init__()
        self.config = config
        self.loop_topology = loop_topology
        
        self.vocabulary = Vocabulary(
            config["word_counts_json"],
            min_count=config["vocab_min_count"]
        )
        self.vocab_size = len(self.vocabulary.node_ids)
        self.x_text,self.y_label = DataReader(input_file=input_file,delim_space=use_word,overfit=overfit)
        self.G = graph
        self.edge_size = self.G.number_of_edges()
        self.edge2id = edge2id

    def __len__(self):
        return len(self.x_text)
    def __getitem__(self, index):
        item = {}

        x = self.x_text[index]
        x = self.vocabulary.to_indices(x)
        x = x[: self.config["max_sequence_length"]]
        y_label = self.y_label[index]
        if self.loop_topology:
            graph_path,outer_edges = GraphWalk.in_degree_edges(self.G,x,self.edge2id)
            x_path,x_path_len,path_mask = pad_sentence(graph_path,self.config["max_sequence_length"]*2,self.vocabulary)
            oedges,oedges_len,oedges_mask = pad_matrix(outer_edges,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        else:
            graph_path,outer_edges = GraphWalk.random_sampling_edges(self.G,x,self.edge2id)
            x_path,x_path_len,path_mask = pad_sentence(graph_path,self.config["max_sequence_length"]*2,self.vocabulary)
            oedges,_ = pad_matrix(outer_edges,self.config["max_sequence_length"],self.config["select_indegree_num"],self.vocabulary)
        #x,xlen = pad_sequences_(x,self.config["max_sequence_length"],self.vocabulary)
        '''print("x_path:{}".format(x_path))
        print("x_path_len:{}".format(x_path_len))
        print("y:{}".format(y_label))
        print("oedges:{}".format(oedges))'''
        item["x_path"] = x_path.long()
        item["y"] = torch.tensor(y_label).long()
        item["xlen"] = torch.tensor(x_path_len).long()
        item["outer_edges"] = oedges.long()
        item["path_mask"] = path_mask.long()
        item["oedges_mask"] = oedges_mask.long()
        return item

    def dev_batch(self):
        x = self.x_tensor[self.indices[self.idx_dev:self.idx_dev+self.batch_size]]
        y = self.y_label[self.indices[self.idx_dev:self.idx_dev+self.batch_size]]
        if self.idx_dev+2*self.batch_size >= len(self.x_tensor):
            self.idx_dev = int(len(self.x_tensor)*(1.0 - self.param_dict["validation"]))
            np.random.shuffle(self.indices[self.idx_dev:])
        else:self.idx_dev += self.batch_size
        if self.loop_topology:
            x_edges = SuperData.in_degree_edges(self,x)
        else:
            x_edges = SuperData.random_sampling_edges(self,x)
        return x,y,x_edges

"""
data handle for predicting
"""
class Data():
    def __init__(self,graph_path='',data_file='',noise=0):
        super().__init__()
        self.noise_k = noise
        self.restore(graph_path)
        self.load(data_file)
        self.idx_train = 0

    def restore(self,output_path):
        self.G = nx.read_gpickle(os.path.join(output_path,"termgraph.pkl"))
        with open(os.path.join(output_path,"word2id.pkl"), 'rb') as input:self.word2id = pickle.load(input)
        with open(os.path.join(output_path,"edge2id.pkl"), 'rb') as input:self.edge2id = pickle.load(input)
        with open(os.path.join(output_path,"config.pkl"),'rb') as input:self.param_dict = pickle.load(input)
        #print("config:{}".format(self.param_dict))
        self.vocab = list(self.word2id.keys())

    def sample(self,original_list):
        if self.noise_k <= 0:return original_list
        sample_word = np.random.choice(self.vocab,self.noise_k)
        original_list.extend(sample_word)
        return original_list

    def load(self,input_file):
        self.x_tensor = list()
        self.y_label = list()
        with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
        for line in data:
            tokens = line.strip().split('\t')
            label = int(tokens[0])
            self.y_label.append(label)
            if self.param_dict["delim_space"]:
                sent_seg = self.sample(tokens[1].strip().split(' '))
            else:
                sent_seg = self.sample(list(tokens[1].strip()))
            sent_list = []
            for item in sent_seg:
                wordid = self.word2id.get(item)
                if wordid is None: wordid = 0
                sent_list.append(wordid)
            self.x_tensor.append(sent_list)
        self.x_tensor = pad_sequences(self.x_tensor, maxlen=self.param_dict["sequence_length"],dtype='int32', padding='post', truncating='post', value=0)
        self.num_batches = int(len(self.x_tensor)/self.param_dict["batch_size"])
        assert len(self.x_tensor) == len(self.y_label)

    def next_batch(self):
        x = self.x_tensor[self.idx_train:self.idx_train+self.param_dict["batch_size"]]
        y = self.y_label[self.idx_train:self.idx_train+self.param_dict["batch_size"]]
        if self.param_dict["loop_topology"]:
            simlex = SuperData.find_loop_simplices(self,x)
            x_edges = SuperData.in_degree_edges(self,x)
        else:
            x_edges = SuperData.random_sampling_edges(self,x)
            simlex = SuperData.random_sampling_simplices(self,x)	
        self.idx_train += self.param_dict["batch_size"]
        return x,y,x_edges,simlex

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

    def reset_batch(self):
        np.random.shuffle(self.indices)
        self.idx_train = 0
        self.idx_dev = int(len(self.x_tensor)*(1.0 - self.param_dict["validation"]))

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

    def dev_batch(self):
        x = self.x_tensor[self.indices[self.idx_dev:self.idx_dev+self.batch_size]]
        y = self.y_cate[self.indices[self.idx_dev:self.idx_dev+self.batch_size]]
        #simplices = self.simplex(x)
        #links = self.indegree_edge(x)
        if self.loop_topology:
            simplices = SuperData.find_loop_simplices(self,x)#self.simplex(x)
            links = SuperData.in_degree_edges(self,x)
        else:
            links = SuperData.random_sampling_edges(self,x)
            simplices = SuperData.random_sampling_simplices(self,x) 
        if self.idx_dev+2*self.param_dict["batch_size"] >= len(self.x_tensor):
            self.idx_dev = int(len(self.x_tensor)*(1.0 - self.param_dict["validation"]))
            np.random.shuffle(self.indices[self.idx_dev:])
        else:self.idx_dev += self.param_dict["batch_size"]
        return x,y,links,simplices

    def build_network(self):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.node_ids)
        self.G.add_edges_from(self.edges)
        self.edge_size = self.G.number_of_edges()

    def save(self,output_path="./"):
        SuperData.save(self,output_path)

if __name__ == "__main__":

    pass

