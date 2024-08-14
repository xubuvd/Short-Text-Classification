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
def set_ax(ax,plt,xlabel="epochs",ylabel="Loss",fontsize=16):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)

if __name__ == "__main__":
    args = parser()

    #plt.figure(figsize=(16,9))
    #plt.style.use('fivethirtyeight')
    plt.style.use('bmh')
    ax = plt.subplot(1,1,1)

    #X,Acc = get_data('./hlog/log_MR_2022031703_34gpu1_20ep64bs.log')
    """
    10：Test Loss:  0.35,  Test Acc: 96.60%
    15：Test Loss:  0.35,  Test Acc: 96.63%
    20：Test Loss:  0.35,  Test Acc: 96.43%
    25：Test Loss:  0.35,  Test Acc: 96.33%
    30：Test Loss:  0.35,  Test Acc: 96.57%
    35：Test Loss:  0.35,  Test Acc: 96.55%
    40：Test Loss:  0.35,  Test Acc: 96.63%
    45：Test Loss:  0.35,  Test Acc: 96.50%
    50：Test Loss:  0.35,  Test Acc: 96.70%
    55：Test Loss:  0.35,  Test Acc: 96.42%
    60：Test Loss:  0.35,  Test Acc: 96.53%
    """

    X =   [0,5,    10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75]
    Acc = [94.41,  96.53,96.44,96.52,96.59,96.42,96.50,96.54,96.44,96.68,96.66,96.42,96.42,96.43,96.42,96.43]
    plt.plot(X,Acc,'-',linewidth = 1.5, color = 'darkred', marker='.', markersize=10, label="")
    plt.xticks(X,size=14)
    plt.yticks([94.4,95.0,95.5,96.0,96.5],size=14)
    #plt.xticks(np.arange(-5, 5, 0.5),fontproperties = 'Times New Roman', size = 10)
    #plt.yticks(np.arange(-2, 2, 0.3),fontproperties = 'Times New Roman', size = 10)

    set_ax(ax,plt, xlabel="K", ylabel="Acc. on DanMu test set",fontsize=22)
    # style
    #plt.style.use(u'seaborn-darkgrid')
    plt.grid(True)
    plt.legend(loc='best', ncol=1,frameon=False,facecolor='lightgray')
    #plt.legend(loc='upper right')
    #plt.title("VisDial v1.0 val",fontsize=18,color='black')
    output_file = "../MySubmission/WSDM2023/pathwalk_ablation_effect_K.jpg"
    if os.path.exists(output_file):os.remove(output_file)
    plt.savefig(output_file,format='jpg',dpi = 1000, bbox_inches='tight')
    plt.show()

