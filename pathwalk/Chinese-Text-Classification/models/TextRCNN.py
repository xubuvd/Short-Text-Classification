# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRCNN'
        self.train_path = '../data/' + dataset + '/data/train.txt'                                # 训练集
        self.dev_path = '../data/' +dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = '../data/' +dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            '../data/' +dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = '../data/' +dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = '../data/' +dataset + '/saved_dict/'        # 模型训练结果
        if False == os.path.exists(self.save_path): os.mkdir(self.save_path)
        self.save_path = self.save_path + self.model_name + '.ckpt'
        self.log_path = '../data/' +dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('../data/' +dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 10                                 # 若超过10 epochs效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 10                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.005                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 300                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数

'''Recurrent Convolutional Neural Networks for Text Classification'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes,bias=True)
        )

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64,20,300]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out

