#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import argparse
from random import randint, random
from math import floor

def fisher_yates_shuffle(the_list):
    list_range = range(0, len(the_list))
    for i in list_range:
        j = randint(list_range[0], list_range[-1])
        the_list[i], the_list[j] = the_list[j], the_list[i]
    return the_list

def fisher_yates_shuffle_improved(the_list):
    amnt_to_shuffle = len(the_list)
    while amnt_to_shuffle > 1:
        i = int(floor(random() * amnt_to_shuffle))
        amnt_to_shuffle -= 1
        the_list[i], the_list[amnt_to_shuffle] = the_list[amnt_to_shuffle], the_list[i]
    return the_list

def shuffle(k,input_file,train_file,test_file):
	data = list()
	class_dict = dict()
	with codecs.open(input_file,'rb','utf-8') as f:
		for line in f:
			tokens = line.strip().split('\t')
			label = tokens[0].strip()
			class_dict.setdefault(label,0)
			class_dict[label] += 1
			data.append(line.strip())
	class_list = sorted(class_dict.items(), key=lambda d: d[1], reverse=True)
	
	data_shuffled = fisher_yates_shuffle_improved(data)
	
	if os.path.exists(train_file):os.remove(train_file)
	train = codecs.open(train_file, "w", "utf-8")
	if os.path.exists(test_file):os.remove(test_file)
	test = codecs.open(test_file, "w", "utf-8")

	train_cnt = 1.0*len(data)*float(k)
	cnt = 0
	for item in data_shuffled:
		if cnt <= train_cnt:
			train.write(item+"\n")
		else:
			test.write(item+"\n")
		cnt += 1
	train.close()
	test.close()

def load(input_file):
	datalist = list()
	with codecs.open(input_file,'rb','utf-8') as f:
		data = f.readlines()
		datalist = [line.strip() for line in data if line.strip()]
	return datalist

def shuffle_data(input_file):
	data = load(input_file)
	data = fisher_yates_shuffle_improved(data)
	for line in data:
		print("{}".format(line))

def parser():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--fold', dest='k', default='0.85',help='k-fold validition')
	parser.add_argument('--input', dest='input_file', default='',help='input: feature samples file')
	parser.add_argument('--train', dest='train_file', default='',help='output: for training')
	parser.add_argument('--test', dest='test_file', default='',help='output: for valdition')
	return parser.parse_args()


if __name__ == "__main__":
	args = parser()
	#shuffle(args.k,args.input_file,args.train_file,args.test_file)
	shuffle_data(args.input_file)

