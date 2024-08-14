import torch
import torch.nn as nn
import numpy as np
from dynamic_rnn import DynamicRNN

bs=32
seq=10
in_size=300
inputs_numpy = np.random.random((bs,seq,in_size))
inputs = torch.from_numpy(inputs_numpy).to(torch.float32)
inputs_len = torch.from_numpy(np.random.randint(low=1,high=seq,size=(bs,))).to(torch.int32)
print("inputs:",inputs.shape)
print("inputs_len:",inputs_len.shape)

hidden_size = 300
lstm = DynamicRNN(nn.LSTM(in_size,hidden_size, batch_first=True, num_layers=2,bidirectional=True))
#lstm = nn.LSTM(300, 128, batch_first=True, num_layers=2,bidirectional=True)
output, (hn, cn) = lstm(inputs,inputs_len)
print("output:",output.shape)#[64,32,128+128]
print("hn:",hn.shape)#[4,64,128]
print("cn:",cn.shape)#[4,64,128]

#output_last = output[:,-1,:3]
#hn_last = hn[2]
#print(output_last.eq(hn_last))#True



