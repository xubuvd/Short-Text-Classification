# -*- coding:utf-8 -*-
import os,sys
import codecs
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--i', dest='input_file', default='',help='input data file')
    parser.add_argument('--o', dest='output_file', default='',help='output file')
    return parser.parse_args()

def get_data(input_file,idx=1):
    '''
    epoch:1	Acc:0.6243489583333334	train_loss:0.6700989603996277	dev_loss:0.6239965558052063	lr:0.004965811965811966
    '''
    def get_mean_rank(strval,key,loss,idx):
        pos = strval.split(key)
        loss.append(float(pos[idx].strip().split(':')[-1]))
    with codecs.open(input_file,'rb','utf-8') as f: data = f.readlines()
    metrics = []
    for line in data:
        line = line.strip()
        if line.find("Acc:") == -1 or line.find("epoch:") == -1:continue
        get_mean_rank(line,"\t",metrics,idx=1)
    xlen = len(metrics)
    X = np.arange(1,xlen+1)
    return X,metrics

'''
blue deepskyblue red m  
'''
def set_ax(ax,plt,xlabel="epochs",ylabel="Loss"):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.ylabel(ylabel,fontsize=16)
    plt.xlabel(xlabel,fontsize=16)

if __name__ == "__main__":
    args = parser()

    plt.figure(figsize=(16,9)) 
    ax = plt.subplot(1,1,1)

    X=range(1,101)
    Acc=[0.97]*100
    plt.plot(X,Acc,'-',linewidth = 1.0, color = 'darkred',marker='' ,label="97")

    X,Acc = get_data('./HistoryLog/log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_45.log')
    plt.plot(X,Acc,'-',linewidth = 1.0, color = 'gray',marker='' ,label="base")
    
    X,Acc = get_data('./HistoryLog/log_PathWalk_2022061101_DanMu_new.log')
    plt.plot(X,Acc,'-',linewidth = 1.0, color = 'blue',marker='' ,label="best>97.")

    X,Acc = get_data('./log_PathWalk_2022061301_DanMu_new.log')
    plt.plot(X,Acc,'-',linewidth = 1.0, color = 'red',marker='' ,label="new")

    set_ax(ax,plt,ylabel="Acc. on dev")
    # style
    plt.style.use(u'seaborn-darkgrid')
    plt.grid(False)
    plt.legend(loc='best', ncol=1,frameon=False,facecolor='lightgray')
    #plt.legend(loc='upper right')
    #plt.title("VisDial v1.0 val",fontsize=18,color='black')
    output_file = "./imgs/path_202206.jpg"
    if os.path.exists(output_file):os.remove(output_file)
    plt.savefig(output_file,format='jpg',dpi = 600, bbox_inches='tight')
    plt.show()

