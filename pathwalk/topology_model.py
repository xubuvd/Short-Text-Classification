# -*- coding:utf-8 -*-
import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

class PathWalkClassificationModel():
	def __init__(self,embedding=128,category=2,batch_size=128,sentence_len=10,l2_lambda=0.2,num_links=5,node_size=10,learning_rate=0.001):
		self.embedding_size = embedding
		self.num_category = category
		self.batch_size = batch_size
		self.sequence_length = sentence_len
		self.l2_regul_lambda = l2_lambda
		self.num_links = num_links
		self.hidden_size = embedding
		self.num_node_link = node_size
		self.learning_rate = learning_rate
		
		np.random.seed(1337)
		tf.set_random_seed(1337)
		self.initializer=tf.random_normal_initializer(mean=0,stddev=0.001)

		self.input_layer()
		self.embedding_layer()
		self.feature_transform()
		self.biLSTM_model()
		self.loss()
		self.reset_unknow()

	def input_layer(self):
		#Input layer
		self.X = tf.placeholder(dtype=tf.int32,shape=[None,self.sequence_length],name='input')
		self.Y = tf.placeholder(dtype=tf.int32,shape=[None],name='label')
		self.AdjLinksTensor = tf.placeholder(dtype=tf.int32,shape=[None,self.sequence_length,self.num_links],name='links')
		self.SimpliceTensor = tf.placeholder(dtype=tf.int32,shape=[None,self.sequence_length,self.num_links],name='nbrs')

	def embedding_layer(self):
		#Embedding layer & Parameters for learning
		self.embeddings = tf.get_variable(name="embeddings",shape=[self.num_node_link, self.embedding_size],initializer=self.initializer,trainable=True)

	def feature_transform(self):
		#sequence text data feature
		feature_vector = tf.nn.embedding_lookup(self.embeddings, self.X) #(batch_size, sequence_length, embedding_size)
		
		#in-degree links feature
		AdjLinksArray = tf.concat(tf.unstack(self.AdjLinksTensor,num=self.batch_size,axis=0),axis=0)
		feat_adjLinks = tf.nn.embedding_lookup(self.embeddings,AdjLinksArray) #(batch_size*sequence_length,indegree_link_num,embedding_size)
		feat_adjLinksSum = tf.reduce_sum(feat_adjLinks,axis=1,keepdims=False) #(batch_size*sequence_length,embedding_size)
		feat_adjLinksSumTensor = tf.reshape(feat_adjLinksSum, [self.batch_size,-1,self.embedding_size]) #(batch_size,sequence_length,embedding_size)

		#simplical complex feature
		complex_array = tf.concat(tf.unstack(self.SimpliceTensor,num=self.batch_size,axis=0),axis=0)
		feat_complex = tf.nn.embedding_lookup(self.embeddings,complex_array)
		feat_complexSum = tf.reduce_sum(feat_complex,axis=1,keepdims=False)
		feat_complexSumTensor = tf.reshape(feat_complexSum, [self.batch_size,-1,self.embedding_size])
		
		tf.add_to_collection('input_tensor', feat_complexSumTensor)
		tf.add_to_collection('input_tensor', feature_vector)
		tf.add_to_collection('input_tensor', feat_adjLinksSumTensor)
		self.node_link_embedding_sum = tf.add_n(tf.get_collection('input_tensor'))
		#self.node_link_embedding_sum = tf.add(feat_adjLinksSumTensor,feature_vector,feat_complexSumTensor) #(batch_size,sequence_length,embedding_size)

	def biLSTM_model(self):
		lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
		lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
		outputs,states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,inputs=self.node_link_embedding_sum, dtype=tf.float32,time_major=False)
		output_rnn = tf.concat(outputs,axis=2)
		output_rnn_last = output_rnn[:,-1,:]#(batch_size,embedding_size*2)

		self.W = tf.get_variable(name="softmax_W",shape=[self.hidden_size*2, self.num_category],initializer=self.initializer,trainable=True)
		self.b = tf.get_variable(name="softmax_b",shape=[self.num_category],initializer=self.initializer,trainable=True)
		self.y_hat = tf.nn.softmax(tf.matmul(output_rnn_last,self.W) + self.b, name="predictions")

	def loss(self):
		#Loss
		onehot_labels = tf.one_hot(indices=self.Y,depth=self.num_category,on_value=1.0,off_value=0.0)
		self.cross_entropy = -tf.reduce_mean(onehot_labels*tf.log(self.y_hat),axis = 0) + self.l2_regul_lambda*tf.nn.l2_loss(self.W)
	
		#Training
		self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
		#Accurrency
		self.accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_hat,1),tf.argmax(onehot_labels,1)),tf.float32),name="accurrency")
	
	def reset_unknow(self):
		self.reset_bias_tensor = tf.scatter_update(self.embeddings, [0], tf.constant(0.0,shape=[1,self.embedding_size],dtype=tf.float32),name="reset_bias_tensor")

