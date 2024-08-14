# -*- coding:utf-8 -*-
import os
import argparse
import glob
import codecs
import pickle
import numpy as np
import collections
from tqdm import tqdm
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

def LoadStopWords(input_file):
    datalines = load(input_file)
    stop_words_set = set()
    for line in datalines:
        line = line.strip()
        if len(line) < 1: continue
        if line not in stop_words_set: stop_words_set.add(line)
    return stop_words_set

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

def stat(input_file):
    data = load(input_file)
    lines = 0
    lsum = 0
    len_list = list()
    max_sent_len = 0
    min_sent_len = 100
    for line in data:
        tokens = line.strip().split('\t')
        if input_file.find("CVQD") != -1 or input_file.find("DanMu") != -1:
            sent = list(tokens[1].strip())
        else:
            sent = tokens[1].strip().split(' ')
        lsum += len(sent)
        lines += 1
        len_list.append(len(sent))
        if len(sent) > max_sent_len: max_sent_len = len(sent)
        if len(sent) < min_sent_len: min_sent_len = len(sent)
    print("avgWords:{}".format(1.0*lsum/lines))
    lensent = np.array(len_list)
    print("avg:{}\tstd:{}".format(np.mean(lensent),np.std(lensent)))
    print("max_sent_len:",max_sent_len)
    print("min_sent_len:",min_sent_len)

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
        if len(text) < 2 or len(label) < 2: 
            print(line)
            continue
        if text in lines_set: continue
        lines_set.add(text)
        ofile.write("{}\t{}\n".format(label,text))
    ofile.close()

def toLabel(input_file,label_file):
    dataset = load(input_file)
    labels = load(label_file)
    label2id_dict = {label.strip():index for index,label in enumerate(labels)}
    print(label2id_dict)
    label_count_stat = dict()
    output_file = "1.txt"
    if os.path.exists(output_file):os.remove(output_file)
    ofile = codecs.open(output_file, "w", "utf-8")
    for line in dataset:
        tokens = line.strip().split('\t')
        label = tokens[0].strip()
        text = tokens[1].strip()
        label_id = label2id_dict[label]
        #print("label:",label,"label_id",label_id)
        ofile.write("{}\t{}\n".format(label_id,text))
        if label_id in label_count_stat: label_count_stat[label_id] += 1
        else: label_count_stat[label_id] = 1
    ofile.close()
    print(label_count_stat)


def MergerSeparateData(input_dir=""):

    input_dir = "./txt_sentoken/pos/"
    pos_files = sorted(glob.glob(os.path.join(input_dir,"*.txt")))
    output_file = "mreviews.txt"
    if os.path.exists(output_file):os.remove(output_file)
    ofile = codecs.open(output_file, "w", "utf-8")
    for file in tqdm(pos_files,total=len(pos_files)):
        filename = os.path.basename(file)
        doc_id = filename.strip().split('_')[-1].split('.')[0].strip()
        print(file,filename,doc_id)
        for line in open(file):
            line = line.strip()
            if len(line) < 2: continue
            ofile.write("pos\t{}\t{}\n".format(doc_id,line))

    input_dir = "./txt_sentoken/neg/"
    neg_files = sorted(glob.glob(os.path.join(input_dir,"*.txt")))
    for file in tqdm(neg_files,total=len(neg_files)):
        filename = os.path.basename(file)
        doc_id = filename.strip().split('_')[-1].split('.')[0].strip()
        print(file,filename,doc_id)
        for line in open(file):
            line = line.strip()
            if len(line) < 2: continue
            ofile.write("neg\t{}\t{}\n".format(doc_id,line))
    ofile.close()


def parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', dest='input_file', default='',help='input: feature samples file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    #stat(args.input_file)
    #remove_duplicate("raw_video_query.data")
    #toLabel("raw/CVQD_label_text.data","data/class.txt")
    MergerSeparateData()


