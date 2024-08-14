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
    epoch:9	loss:58.71915817260742	lr:0.00010000000000000002
    '''
    def get_mean_rank(strval,key,loss):
        pos = strval.split(key)
        loss.append(float(pos[idx].strip().split(':')[-1]))
    with codecs.open(input_file,'rb','utf-8') as f: data = f.readlines()
    Loss = []
    for line in data:
        if line.find("loss:") == -1:continue
        get_mean_rank(line,"\t",Loss)
    xlen = len(Loss)
    print(Loss)
    X = np.arange(1,xlen+1)
    return X,Loss

'''
blue deepskyblue red m  
'''
def set_ax(ax,plt,xlabel="iters",ylabel="Loss"):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.ylabel(ylabel,fontsize=16)
    plt.xlabel(xlabel,fontsize=16)

if __name__ == "__main__":

    args = parser()

    #plt.figure(figsize=(16,8)) 
    ax = plt.subplot(1,2,1)
    #X,Loss = get_data('./HistoryLog/log_A18_v6.5.6_gen_2020121702.log',idx=1)
    #X,Loss = get_data('./HistoryLog/log_A18_v6.5.5_2020121603.log',idx=2)
    #X,Loss = get_data('./HistoryLog/log_A18_v6.5.6_gen_2020121703.log',idx=1)
    #plt.plot(X,Loss,'-',linewidth = 1.0, color = 'red')#,marker='' ,label="A18_v6.5.6_Loss")
    #set_ax(ax,plt,ylabel="Loss")

    ax = plt.subplot(1,1,1)
    X,Loss = get_data('./HistoryLog/lr.log',idx=2)
    plt.plot(X,Loss,'.',linewidth = 1.0, color = 'blue')#,marker='' ,label="A18_v6.5.6_Lr")
    set_ax(ax,plt,ylabel="Learning rate")

    # style
    plt.style.use(u'seaborn-darkgrid')
    plt.grid(False)

    plt.legend(loc='best', ncol=1,frameon=False,facecolor='lightgray')
    #plt.legend(loc='upper right')
    #plt.title("VisDial v1.0 val",fontsize=18,color='black')
    output_file = "./imgs/visdial_20202230.png"
    if os.path.exists(output_file):os.remove(output_file)
    plt.savefig(output_file,format='png',dpi = 600, bbox_inches='tight')
    plt.show()

