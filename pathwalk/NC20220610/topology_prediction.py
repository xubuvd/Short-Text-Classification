# -*- coding:utf-8 -*-
import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
from graph_batch import Data

#set seed for reproducible results
np.random.seed(1337)
tf.set_random_seed(1337)

#KuaiShou Model Path
tf.flags.DEFINE_string("save_model_path","../topology_model/pathwalk","save path for tensorflow model")
tf.flags.DEFINE_string("save_graph_path","../topology_graph/","save path for graph model")
tf.flags.DEFINE_string("test_files_path","../data/fortest.txt","test datasets")

#review_polarity Model Path
#tf.flags.DEFINE_string("save_model_path","../review_polarity_model/pathwalk","save path for tensorflow  model")
#tf.flags.DEFINE_string("save_graph_path","../review_polarity_graph/","save path for graph  model")#
#tf.flags.DEFINE_string("test_files_path","../data/review_polarity/fortest.txt","test datasets")   

#VideoQuery Model Path
#tf.flags.DEFINE_string("save_model_path","../video_model/pathwalk","save path for tensorflow  model")
#tf.flags.DEFINE_string("save_graph_path","../video_graph/","save path for graph  model")
#tf.flags.DEFINE_string("test_files_path","../data/VideoQuery/fortest_seg.txt","test datasets")   


FLAGS = tf.flags.FLAGS

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(FLAGS.save_model_path+'.meta')
	saver.restore(sess,FLAGS.save_model_path)

	graph = tf.get_default_graph()
	X = graph.get_tensor_by_name("input:0")
	Y = graph.get_tensor_by_name("label:0")
	L = graph.get_tensor_by_name("links:0")
	S = graph.get_tensor_by_name("nbrs:0")
	pred = graph.get_tensor_by_name("predictions:0")
	acc = graph.get_tensor_by_name("accurrency:0")

	for noise_num in range(0,10):
		data = Data(graph_path=FLAGS.save_graph_path, data_file=FLAGS.test_files_path, noise=noise_num)
		#print("num_batches:{}".format(data.num_batches))
		acc_list = list()
		for i in range(data.num_batches):
			x,y,x_edge,simlex = data.next_batch()
			feed = {X:x,L:x_edge,Y:y,S:simlex}
			res = sess.run(pred,feed_dict=feed)
			acc_res = sess.run(acc,feed_dict=feed)
			#print("{}, {:.4f}".format(np.argmax(res,axis=1),acc_res))
			acc_list.append(acc_res)
		print("noise_add:{}\t{}\t{:.4f}".format(noise_num,acc_list,np.mean(acc_list)))

