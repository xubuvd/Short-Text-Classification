# -*- coding:utf-8 -*-
import os
import sys
import torch
from torch import nn
import numpy as np
from layer import GELU
from dynamic_rnn import DynamicRNN
from torch.nn import functional as F

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
            nn.Linear(config["lstm_hidden_size"]*2,num_class,bias=True)
        )

    def forward(self,batch):
        '''
        Input parameters:
            x, bag of words, [batch_size,1,max_sequence_length]
            x_len, [batch_size]
        '''
        x = batch["x"].squeeze(1)#[batch_size,max_seq_len]
        xlen = batch["xlen"]#[batch_size]
        x_embed = self.word_embed(x)#[32,10,512]
        batch_size, max_sequence_length, embedding_size = x_embed.shape

        output_step, (h, c) = self.TextBiLSTM(x_embed,xlen)
        logit = self.encoder_linear(torch.cat([h[2],h[3]],1))#[32,num_class]
        prob = F.softmax(logit,dim=1)#[32,num_class]
        return prob

