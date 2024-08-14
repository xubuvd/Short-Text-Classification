import torch
import numpy as np
import random

max_sequence_length=3
select_indegree_num=4

node_mask = torch.full(
    (max_sequence_length,select_indegree_num),
    fill_value=0
)
#print(node_mask)

sequences = [[1,2,3,4],[5,6,7]]
for i in range(len(sequences)):
    sequences[i] = sequences[i][:select_indegree_num]
    node_mask[i][:len(sequences[i])] = 1
#print(sequences)
#print(node_mask)


inputs_numpy = np.random.random((2,4,3))
inputs = torch.from_numpy(inputs_numpy).to(torch.float32)
#print(inputs)

a = torch.zeros_like(inputs)
for i in range(4-1):
    a[:,i,:] = inputs[:,i+1,:] - inputs[:,i,:]
#print(a)

a = [['a','b','c'],['d','e','f','g']]
b = [[1,2,3],[6,7,8,9]]
for i in range(3):
    for i in range(len(a)):
        lzip = list(zip(a[i],b[i]))
        random.shuffle(lzip)
        a[i][:],b[i][:] = zip(*lzip)

for j in range(0,0):
    print("j:",j)










