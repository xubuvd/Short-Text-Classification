# -*- coding:utf-8 -*-
import os
import sys
import torch
from torch import nn
import numpy as np
from layer import GELU
from dynamic_rnn import DynamicRNN
from torch.nn import functional as F
#from pytorch_pretrained import BertModel, BertTokenizer
from self_attention import UnitSelfAttention

class AttentionLayer(nn.Module):
    def __init__(self,config):
        super(AttentionLayer,self).__init__()
       
        self.linear1 = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["embedding_size"],config["embedding_size"],bias=True),
            #nn.LayerNorm(config["embedding_size"]),
            #GELU()
        )
        self.linear2 = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["embedding_size"]*2,1,bias=True),
        )

    def forward(self,x,y,select_indegree_num):
        '''
        x, [32,10,512]
        y, [32*10,36,512]
        select_indegree_num, 36 as default
        '''
        batch_size, max_sequence_length, embedding_size = x.shape

        #first attn layer
        x_ = self.linear1(x); y_ = self.linear1(y)
        x_ = x_.reshape(batch_size*max_sequence_length,-1)#[32*10,512]
        x_y_cat = torch.cat([x_.unsqueeze(1).repeat(1,select_indegree_num,1),y_],dim=-1)#[32*10,36,512+512]
        logit = self.linear2(x_y_cat)#[32*10,36,1]
        prob = F.softmax(logit,dim=1)#[32*10,36,1]
        a = torch.mul(y,prob).sum(1).reshape(batch_size,max_sequence_length,-1)#[32,10,512]
        return a

class PathWalkModel(nn.Module):
    def __init__(self,config,node_size=10,vocabulary=None,num_class=2):
        super(PathWalkModel,self).__init__()
        self.config = config
        
        self.word_embed = nn.Embedding(
            node_size,
            config["embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )

        self.TextBiLSTM = DynamicRNN(
            nn.LSTM(
                config["embedding_size"],
                config["lstm_hidden_size"],
                config["lstm_num_layers"],
                batch_first=True,
                bidirectional=config['txt_bidirectional'],
                dropout=config["dropout"]
            )
        )

        self.encoder_linear = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["lstm_hidden_size"]*2,1,bias=True)
        )
        self.pred_linear = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["lstm_hidden_size"]*2,num_class,bias=True)
        )
        self.InEdgeProj = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["embedding_size"]*3,config["embedding_size"],bias=True),
            nn.LayerNorm(config["embedding_size"]),
            GELU()
        )
        
        self.OuterNodeEdgeFuse = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["embedding_size"]*3,config["embedding_size"],bias=True),
            nn.LayerNorm(config["embedding_size"]),
            GELU()
        )
        self.NodeEdgeFuse = nn.Sequential(
            nn.Dropout(config["dropout_fc"]),
            nn.Linear(config["embedding_size"]*3,config["embedding_size"],bias=True),
            nn.LayerNorm(config["embedding_size"]),
            GELU()
        )
        self.attn = AttentionLayer(config)
        self.self_attn = UnitSelfAttention(hidden_size=config["embedding_size"], dropout=config["dropout_fc"])

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
        x_edges = batch["in_edges"].squeeze(1)#[batch_size,select_indegree_num]
        inedge_mask = batch["inedge_mask"]#[batch_size,max_seq_len,select_indegree_num]
        x_mask = batch["x_mask"].squeeze(1)#[batch_size,max_seq_len]
        prefix_node = batch["prefix_node"]#[batch_size,max_seq_len,select_indegree_num]
        edge1 = batch["edge1"]#[batch_size,max_seq_len,select_indegree_num]
        outer_node = batch["outer_node"]#[batch_size,max_seq_len,select_indegree_num]
        edge2 = batch["edge2"]
        successor_node = batch["successor_node"]

        x_embed = self.word_embed(x)#[32,10,512]
        batch_size, max_sequence_length, embedding_size = x_embed.shape
        output_step, (h, c) = self.TextBiLSTM(x_embed,xlen)
        
        x_edges_embed = self.word_embed(x_edges)#[32,10,512] 
        subtraction = torch.zeros_like(output_step)#[32,10,1024]
        for i in range(max_sequence_length - 1):
            subtraction[:,i,:] = output_step[:,i+1,:] - output_step[:,i,:]
        # Update edge representaion
        x_edges_embed = self.InEdgeProj(
            torch.cat([x_edges_embed,subtraction],dim=-1)
        )#[32,10,512] 
        # graph path based feature
        x_sum_embed = self.NodeEdgeFuse(
            torch.cat([output_step,x_edges_embed],dim=-1)
        )##[32,10,512+512]->[32,10,512]
        #output_step, (h, c) = self.TextBiLSTM(x_sum_embed,xlen)
        #output_step:torch.Size([32, 10, 512+512]
        #h:torch.Size([4, 32, 512]), h[0]-1th forward, h[1]-1th backward, h[2]-2th forward, h[3]-2th backward
        #c:torch.Size([4, 32, 512])


        # topology based neighbors
        '''
        prefix node: A
        outer node: E
        successor node: B
        edge1: (A,E)
        edge2: (E,B)
        A -> B -> C
        A -> E -> B
        '''

        _, _, select_indegree_num = prefix_node.shape
        #prefix_node = prefix_node.reshape(batch_size*max_sequence_length,select_indegree_num)
        edge1 = edge1.reshape(batch_size*max_sequence_length,select_indegree_num)
        outer_node = outer_node.reshape(batch_size*max_sequence_length,select_indegree_num)
        edge2 = edge2.reshape(batch_size*max_sequence_length,select_indegree_num)
        #successor_node = successor_node.reshape(batch_size*max_sequence_length,select_indegree_num)

        prefix_node_embed = self.word_embed(prefix_node)#[32*10,36,512]
        edge1_embed = self.word_embed(edge1)#[32*10,36,512]
        outer_node_embed = self.word_embed(outer_node)#[32*10,36,512]
        edge2_embed = self.word_embed(edge2)#[32*10,36,512]
        successor_node_embed = self.word_embed(successor_node)#[32*10,36,512]
        
        #edge1_sub = outer_node_embed - prefix_node_embed #[32*10,36,512]
        #edge2_sub = successor_node_embed - outer_node_embed #[32*10,36,512]
       
        # Update representation for outer edge
        #edge1_embed = self.EdgeProj(torch.cat([edge1_embed,edge1_sub],dim=-1))#[32*10,36,512]
        #edge2_embed = self.EdgeProj(torch.cat([edge2_embed,edge2_sub],dim=-1))#[32*10,36,512]
        
        nbhd_sum_embed = self.OuterNodeEdgeFuse(torch.cat([edge1_embed,outer_node_embed,edge2_embed],dim=-1))#[32*10,36,512]
        x_attn_embed = self.attn(x_sum_embed,nbhd_sum_embed,select_indegree_num)#[32,10,512]

        sent_rep = self.self_attn(x_sum_embed,x_mask)#[32,512]
 
        hyb = torch.cat([sent_rep.unsqueeze(1).repeat(1,max_sequence_length,1),x_attn_embed],dim=-1)#[32,10,512+512]
        logit = self.encoder_linear(hyb)#[32,10,1]
        a = F.softmax(logit,dim=1)#[32,10,1]
        nbhd = torch.mul(x_attn_embed,a).sum(1)#[32,512]
        
        res = self.pred_linear(torch.cat([sent_rep,nbhd],1))#[32,num_class]
        prob = F.softmax(res,dim=1)#[32,num_class]
        return prob

