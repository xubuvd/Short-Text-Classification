# -*- coding:utf-8 -*-
import os
import codecs
import pickle
import numpy as np
import collections
from shuffle import fisher_yates_shuffle_improved 
import re

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def load(input_file):
	with codecs.open(input_file,'rb','utf-8') as f:data = f.readlines()
	return data

def listDir(dir_path):
	for txtfile in os.listdir(dir_path):
		txt = os.path.join(dir_path,txtfile)
		yield txt

def read_data(pos_data,pos_dir):
	for txtfile in listDir(pos_dir):
		print(txtfile)
		data = load(txtfile)
		pos_data.extend(data)	

def pre_process():
	pos_dir = "./txt_sentoken/pos"
	pos_data = list()
	read_data(pos_data,pos_dir)
	
	neg_dir = "./txt_sentoken/neg"
	neg_data = list()
	read_data(neg_data,neg_dir)

	print("pos data:{}".format(len(pos_data)))
	print("neg data:{}".format(len(neg_data)))
	
	data = list()
	for line in pos_data:data.append("1"+"\t"+line.strip())
	for line in neg_data:data.append("0"+"\t"+line.strip())
	data = fisher_yates_shuffle_improved(data)

	output_file = "review_polarity.txt"
	if os.path.exists(output_file):os.remove(output_file)
	output = codecs.open(output_file,'wb',encoding='utf-8')
	for line in data:output.write(line+"\n")

def clear():
	label2id = dict()
	label_data = dict()
	data = load("video_query_moview_and_sports.txt")
	for line in data:
		tokens = line.strip().split('\t')
		if len(tokens) != 2:continue
		label = tokens[1].strip()
		if label == "其他":continue
		class_id = len(label2id)
		if label not in label2id:label2id[label] = len(label2id)
		class_id = label2id[label]
		sent = tokens[0].strip()
		if class_id not in label_data:label_data[class_id] = list()
		label_data[class_id].append(sent)
	for label_id in label_data:
		labelsize = len(label_data[label_id])
		data = fisher_yates_shuffle_improved(label_data[label_id])
		cnt = 0
		for item in data:
			print("{}\t{}".format(label_id,item))
			cnt += 1
			if cnt > 100000:break
	#with open("topiclabel2id.pkl", 'wb') as output:pickle.dump(label2id,output)

def stat():
	data = load("train.txt")
	lines = 0
	lsum = 0
	len_list = list()
	for line in data:
		tokens = line.strip().split('\t')
		sent = list(tokens[1].strip())
		lsum += len(sent)
		lines += 1
		len_list.append(len(sent))
	print("avgWords:{}".format(1.0*lsum/lines))
	lensent = np.array(len_list)
	print("avg:{}\tstd:{}".format(np.mean(lensent),np.std(lensent)))

def remove_duplicate(input_file):
    dataset = load(input_file)
    lines_set = set()
    
    output_file = "1.txt"
    if os.path.exists(output_file):os.remove(output_file)
    ofile = codecs.open(output_file, "w", "utf-8")

    for line in dataset:
        line = line.strip()
        if len(line) < 2: 
            print(line)
            continue
        tokens = line.split('\t')
        if len(tokens) != 2:
            print(line)
            continue
        text = tokens[0].strip()
        label = tokens[1].strip()
        if label == "文化": label = "其他"
        if len(text) < 2 or len(label) < 2: 
            print(line)
            continue
        if text in lines_set: continue
        lines_set.add(text)
        ofile.write("{}\t{}\n".format(label,text))
    ofile.close()

if __name__ == "__main__":
	#stat()
    remove_duplicate("raw_video_query.data")

