# -*- coding:utf-8 -*-
import os
import sys
import torch
from torch import nn
import numpy as np
from layer import GELU
from dynamic_rnn import DynamicRNN
from torch.nn import functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class AttentionLayer(nn.Module):
    def __init__(self,config,bert_hidden_size):
        super(AttentionLayer,self).__init__()
        self.linear1 = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.LayerNorm(bert_hidden_size),
            #nn.Dropout(config["dropout_fc"]),# epoch:1	improve Acc:0.5963541666666666 to Acc:0.6022135416666666
            nn.Linear(bert_hidden_size,bert_hidden_size,bias=True),
            nn.Dropout(config["dropout_fc"])
        )
        self.linear2 = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(bert_hidden_size*2,1,bias=True),
        )
        self.fun = nn.Sequential(
            GELU()
        )

    def forward(self,x,y,select_indegree_num):
        '''
        x, [32,10,512]
        y, [32*10,36,512]
        select_indegree_num, 36 as default
        '''
        batch_size, max_sequence_length, embedding_size = x.shape
        x_ = self.linear1(x); y = self.linear1(y)
        x_ = x_.reshape(batch_size*max_sequence_length,embedding_size)#[32*10,512]
        x_y_cat = torch.cat([x_.unsqueeze(1).repeat(1,select_indegree_num,1),y],dim=-1)#[32*10,36,512+512]
        logit = self.linear2(x_y_cat)#[32*10,36,1]
        prob = F.softmax(logit,dim=1)#[32*10,36,1]
        a = torch.mul(y,prob).sum(1).reshape(batch_size,max_sequence_length,embedding_size)#[32,10,512]
        x = self.fun(x.reshape(batch_size,max_sequence_length,embedding_size) + a)
        return x

class MultiHeadAttn(nn.Module):
    def __init__(self,config):
        super(MultiHeadAttn,self).__init__()
        self.num_cross_attns = config["multi_heads"]
        self.has_residual = config['has_residual']
        self.layers = nn.ModuleList([AttentionLayer(config) for _ in range(self.num_cross_attns)])
        self.MLP = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["bert_hidden_size"]*self.num_cross_attns,config["bert_hidden_size"],bias=True),
            GELU()
        )
    def forward(self,x,y,select_indegree_num):
        res_list = []
        for i in range(self.num_cross_attns):
            output = self.layers[i](x,y,select_indegree_num)
            res_list.append(output)
        input_x = torch.cat(res_list,dim=-1)#[32,10,512*multi_heads]
        res = self.MLP(input_x)
        if self.has_residual: res = x + res
        return res

class BertPathWalkModel(nn.Module):
    def __init__(self,config,node_size=10,vocabulary=None,num_class=2):
        super(BertPathWalkModel,self).__init__()
        self.config = config
        self.TextBiLSTM = DynamicRNN(
            nn.LSTM(
                config["bert_hidden_size"],
                config["bert_hidden_size"],
                config["lstm_num_layers"],
                batch_first=True,
                bidirectional=config['txt_bidirectional'],
                dropout=config["dropout"]
            )
        )
        self.bert = BertModel.from_pretrained(config['bert_path'])
        for param in self.bert.parameters(): param.requires_grad = True
        #self.word_embed = nn.Embedding(node_size,config["bert_hidden_size"],padding_idx=vocabulary.PAD_INDEX)
        #self.word_embed = self.bert.embeddings.word_embeddings.weight
        #self.word_embed = nn.Embedding.from_pretrained(self.bert.embeddings.word_embeddings.weight, freeze=False)
        self.encoder_linear = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["bert_hidden_size"]*2,num_class,bias=True)
        )

        self.EdgeProj = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["bert_hidden_size"],config["bert_hidden_size"],bias=True),
            GELU()
        )
        self.OuterNodeEdgeFuse = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["bert_hidden_size"]*3,config["bert_hidden_size"],bias=True),
            GELU(),
            nn.LayerNorm(config["bert_hidden_size"])
        )
        self.NodeEdgeFuse = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["bert_hidden_size"]*2,config["bert_hidden_size"],bias=True),
            GELU(),
            nn.LayerNorm(config["bert_hidden_size"])
        )
        self.attn = AttentionLayer(config,config["bert_hidden_size"])

    def forward(self,batch):
        '''
        Input parameters:
            x, bag of words, [batch_size,1,max_sequence_length]
            x_len, [batch_size]
            x_mask, [batch_size,1,max_sequence_length]
            x_edges, outer_edge, [batch_size, max_sequence_length, select_indegree_num]
            inedge_mask, [batch_size, max_sequence_length, select_indegree_num]
            nbhd_nodes, [batch_size, max_sequence_length, select_indegree_num]
            nbhd_nodes_mask, [batch_size, max_sequence_length, select_indegree_num]
            nbhd_edges, same as nbhd_nodes
            nbhd_edges_mask, same as nbhd_edges
        '''
        x = batch["x"].squeeze(1)#[batch_size,max_seq_len]
        xlen = batch["xlen"]#[batch_size]
        #x_edges = batch["in_edges"].squeeze(1)#[batch_size,select_indegree_num]
        #inedge_mask = batch["inedge_mask"]#[batch_size,max_seq_len,select_indegree_num]
        x_mask = batch["x_mask"].squeeze(1)#[batch_size,max_seq_len] [64, 30]
        prefix_node = batch["prefix_node"]#[batch_size,max_seq_len,select_indegree_num]
        prefix_mask = batch["prefix_mask"]##[batch_size,max_seq_len,select_indegree_num]
        #edge1 = batch["edge1"]#[batch_size,max_seq_len,select_indegree_num]
        outer_node = batch["outer_node"]#[batch_size,max_seq_len,select_indegree_num]
        #edge2 = batch["edge2"]
        successor_node = batch["successor_node"]
        outer_mask = batch["outer_mask"]
        successor_mask = batch["successor_mask"]

        x_embed, _ = self.bert(x, attention_mask=x_mask, output_all_encoded_layers=False)
        #x_embed = self.word_embed(x)#[32,10,512]
        batch_size, max_sequence_length, embedding_size = x_embed.shape
        #x_edges_embed = self.word_embed(x_edges)#[32,10,512]
        
        subtraction = torch.zeros_like(x_embed)#[32,10,512]
        for i in range(max_sequence_length - 1):
            subtraction[:,i,:] = x_embed[:,i+1,:] - x_embed[:,i,:]
        # Update edge representaion
        x_edges_embed = self.EdgeProj(subtraction)#[32,10,512]
        
        # graph path based feature
        x_sum_embed = self.NodeEdgeFuse(
            torch.cat([x_embed,x_edges_embed],dim=-1)
        )##[32,10,512+512]->[32,10,512]
        
        # topology based neighbors
        _, _, select_indegree_num = prefix_node.shape
        prefix_node = prefix_node.reshape(batch_size*max_sequence_length,select_indegree_num)
        #edge1 = edge1.reshape(batch_size*max_sequence_length,select_indegree_num)
        outer_node = outer_node.reshape(batch_size*max_sequence_length,select_indegree_num)
        #edge2 = edge2.reshape(batch_size*max_sequence_length,select_indegree_num)
        successor_node = successor_node.reshape(batch_size*max_sequence_length,select_indegree_num)

        #prefix_mask: [64, 30, 40]
        #prefix_node_embed = self.word_embed(prefix_node)#[32*10,36,512]
        prefix_mask = prefix_mask.reshape(batch_size*max_sequence_length,select_indegree_num)
        prefix_node_embed, _ = self.bert(prefix_node, attention_mask=prefix_mask, output_all_encoded_layers=False)
        #edge1_embed = self.word_embed(edge1)#[32*10,36,512]
        #outer_node_embed = self.word_embed(outer_node)#[32*10,36,512]
        outer_mask = outer_mask.reshape(batch_size*max_sequence_length,select_indegree_num)
        outer_node_embed, _ = self.bert(outer_node, attention_mask=outer_mask, output_all_encoded_layers=False)
        #edge2_embed = self.word_embed(edge2)#[32*10,36,512]
        #successor_node_embed = self.word_embed(successor_node)#[32*10,36,512]
        successor_mask = successor_mask.reshape(batch_size*max_sequence_length,select_indegree_num)
        successor_node_embed, _ = self.bert(successor_node, attention_mask=successor_mask, output_all_encoded_layers=False)
        edge1_sub = outer_node_embed - prefix_node_embed #[32*10,36,512]
        edge2_sub = successor_node_embed - outer_node_embed #[32*10,36,512]
        # Update representaion for outer edge
        edge1_embed = self.EdgeProj(edge1_sub)#[32*10,36,512]
        edge2_embed = self.EdgeProj(edge2_sub)#[32*10,36,512]
        nbhd_sum_embed = self.OuterNodeEdgeFuse(torch.cat([edge1_embed,outer_node_embed,edge2_embed],dim=-1))#[32*10,36,512]
        x_sum_embed = self.attn(x_sum_embed,nbhd_sum_embed,select_indegree_num)#[64, 30, 768]
        #
        output_step, (h, c) = self.TextBiLSTM(x_sum_embed,xlen)
        logit = self.encoder_linear(torch.cat([h[2],h[3]],1))#[32,num_class]
        prob = F.softmax(logit,dim=1)#[32,num_class]
        return prob

