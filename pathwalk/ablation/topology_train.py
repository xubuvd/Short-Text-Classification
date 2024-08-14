# -*- coding:utf-8 -*-
import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
from graph_batch import TopologyData
from topology_model import PathWalkClassificationModel

#set seed for reproducible results
np.random.seed(1337)
tf.set_random_seed(1337)

#Parameters
tf.flags.DEFINE_float("l2_regul_lambda",0.20,"lambda for weight regularization")
tf.flags.DEFINE_integer("embedding_size",128,"embedding size for node and edge representation")
tf.flags.DEFINE_integer("hidden_size",128,"size of hidden in LSTM layer cell")
tf.flags.DEFINE_integer("num_epoch",20,"number of epoch in training")
tf.flags.DEFINE_integer("num_category",2,"number of categories")
tf.flags.DEFINE_integer("batch_size",128,"size of batch for training")
tf.flags.DEFINE_integer("sequence_length",10,"sentence length")
tf.flags.DEFINE_integer("num_links",5,"number of in-degree links involving a term")
tf.flags.DEFINE_integer("save_per_epoch",10,"saved time per epoch iteration during training")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

#KuaiShou Model Path
#tf.flags.DEFINE_string("save_model_path","../topology_model/pathwalk","save path for tensorflow  model")
#tf.flags.DEFINE_string("save_graph_path","../topology_graph/","save path for graph  model")
#tf.flags.DEFINE_string("train_data_path","../data/train_pathWalk.txt","train data file")

#review_polarity Model Path
#tf.flags.DEFINE_string("save_model_path","../review_polarity_model/pathwalk","save path for tensorflow  model")
#tf.flags.DEFINE_string("save_graph_path","../review_polarity_graph/","save path for graph  model")
#tf.flags.DEFINE_string("train_data_path","../data/review_polarity/train_pathWalk.txt","train data file")

#Video Query topic classification
tf.flags.DEFINE_string("save_model_path","../video_model/pathwalk","save path for tensorflow  model")
tf.flags.DEFINE_string("save_graph_path","../video_graph/","save path for graph  model")
tf.flags.DEFINE_string("train_data_path","../data/VideoQuery/train.txt","train data file")

FLAGS = tf.flags.FLAGS

"""
Walk-of-Words model plus in-degree links + simplices complex  for short text classification
"""
sess = tf.InteractiveSession()
data = TopologyData(FLAGS.train_data_path, batch_size=FLAGS.batch_size,\
			sequence_len=FLAGS.sequence_length,\
			indegree=FLAGS.num_links,delim_space=False,
			loop_topology=True)
node_size = data.edge_size + data.vocab_size
print("node size:{}".format(node_size))

model = PathWalkClassificationModel(embedding=FLAGS.embedding_size,
	category=FLAGS.num_category,
	batch_size=FLAGS.batch_size,
	sentence_len=FLAGS.sequence_length,
	l2_lambda=FLAGS.l2_regul_lambda,
	num_links=FLAGS.num_links,
	node_size=node_size)

saver = tf.train.Saver()
if os.path.exists(FLAGS.save_model_path) == False:os.makedirs(FLAGS.save_model_path)
sess.run(tf.global_variables_initializer())

for iter in range(FLAGS.num_epoch):
	for batch_iter in range(data.num_batch):
		sess.run(model.reset_bias_tensor)
		xx,yy,x_edge,x_simplex = data.next_batch()
		feed = {model.X:xx,model.Y:yy,model.AdjLinksTensor:x_edge,model.SimpliceTensor:x_simplex}
		res = sess.run([model.train,model.cross_entropy],feed_dict=feed)
		
		xx_dev, yy_dev,x_edge,x_simplex = data.dev_batch() #toy_sample(node_size, num_category)
		feed = {model.X: xx_dev, model.Y: yy_dev,model.AdjLinksTensor:x_edge,model.SimpliceTensor:x_simplex}
		res_dev = sess.run([model.accur], feed_dict=feed)
		
		print("Epoch:{:3d}\tIteration:{:5d}\tCross_etropy:[{:.4f} {:.4f}]\tValidation Accurrence:{:.4f}".format(iter+1,iter*data.num_batch+batch_iter,res[1][0],res[1][1],res_dev[0]))

	if iter % FLAGS.save_per_epoch == 0:
		sess.run(model.reset_bias_tensor)
		save_path = saver.save(sess, FLAGS.save_model_path)
		print("Model saved in path:{}".format(save_path))
	data.reset_batch()

sess.run(model.reset_bias_tensor)
save_path = saver.save(sess, FLAGS.save_model_path)
print("Tensorflow Model saved in path:{}".format(save_path))

data.save(output_path=FLAGS.save_graph_path)
print("TermGraph saved in path:{}".format(FLAGS.save_graph_path))

