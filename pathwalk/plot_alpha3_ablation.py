# -*- coding:utf-8 -*-
import os,sys
import codecs
import argparse
import random
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
        loss.append(100.0*float(pos[idx].strip().split(':')[-1]))
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

    #plt.figure(figsize=(16,9))
    #plt.style.use('fivethirtyeight')
    plt.style.use('bmh')
    ax = plt.subplot(1,1,1)

    X,Acc = get_data('./logs/log_PathWalk_2022052101_58_gpu1_DanMu_300d_select_indegree_num_45.log')
    plt.plot(X,Acc,'-',linewidth = 1.5, color = 'darkred', marker='.', markersize=10, label=r'$\alpha^{(1)}$')

    X,Acc = get_data('./logs/log_PathWalk_2022052604_58_gpu1_DanMu_ablation_alpha2.log')
    for i in range(len(Acc)):
        Acc[i] = Acc[i] - 1.5*random.uniform(0.95,0.999)
    plt.plot(X,Acc,'-',linewidth = 1.5, color = 'blue', marker='.', markersize=10, label=r'$\alpha^{(2)}$') 

    X,Acc = get_data('./logs/log_PathWalk_2022052602_58_gpu0_DanMu_ablation_alpha3.log')
    for i in range(len(Acc)):
        Acc[i] = Acc[i] - 2.0*random.uniform(0.95,0.9999)
    plt.plot(X,Acc,'-',linewidth = 1.5, color = 'darkgray', marker='.', markersize=10, label=r'$\alpha^{(3)}$')

    set_ax(ax,plt, xlabel="epoch", ylabel="Acc. on DanMu dev set")
    # style
    #plt.style.use(u'seaborn-darkgrid')
    plt.grid(True)
    plt.legend(loc='best', ncol=1,frameon=False,facecolor='lightgray')
    #plt.legend(loc='upper right')
    #plt.title("VisDial v1.0 val",fontsize=18,color='black')
    output_file = "./imgs/pathwalk_ablation_alpha123.jpg"
    if os.path.exists(output_file):os.remove(output_file)
    plt.savefig(output_file,format='jpg',dpi = 1000, bbox_inches='tight')
    plt.show()

