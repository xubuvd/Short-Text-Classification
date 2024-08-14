# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'DPCNN'
        self.train_path = '../data/' + dataset + '/data/train.txt'                                # 训练集
        self.dev_path = '../data/' + dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = '../data/' + dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            '../data/' + dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = '../data/' + dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = '../data/' + dataset + '/saved_dict/'        # 模型训练结果
        if False == os.path.exists(self.save_path): os.mkdir(self.save_path)
        self.save_path = self.save_path + self.model_name + '.ckpt'

        self.log_path = '../data/' + dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('../data/' + dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 10                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 10                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_filters = 250                                          # 卷积核数量(channels数)

'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
			nn.Dropout(config.dropout),
			nn.Linear(config.num_filters, config.num_classes)
		)

    def forward(self, x):
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2: x = self._block(x)
        #x:[64, 250, 2, 1]
        x = x.squeeze()  # [batch_size, num_filters(250)]
        #x:[64, 250, 2]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
