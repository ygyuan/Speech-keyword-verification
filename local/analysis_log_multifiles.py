#!/usr/bin/env python

"""
Analysis log file

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def Analysis_AP(logfile):
    contents = open(logfile).read().split('\n')
    idx_now = -6
    idx = -6
    ap_index=[]
    ap_ave=[]
    ap_pos=[]
    ap_neg=[]
    ap_thr=[]
    for content in contents:
        if content.startswith('Dev AP'):
            idx_now += 5
            ap_index.append(max(idx_now,0))
            content=content[content.find(': ') + 2: ]
            ave,pos,neg,thr=content.split(" ")[:4]
            ap_ave.append(float(ave))
            ap_pos.append(float(pos))
            ap_neg.append(float(neg))
            ap_thr.append(float(thr))
    if idx<0:
        idx=0
    print ap_index[:4],len(ap_index),len(ap_ave)
    print "Reading: ", logfile
    print max(ap_ave),ap_pos[ap_ave.index(max(ap_ave))],ap_neg[ap_ave.index(max(ap_ave))], ap_thr[ap_ave.index(max(ap_ave))]
    return ap_index, ap_ave

def plotimg(ap_ave1, ap_index1, ap_ave2, ap_index2, ap_ave3, ap_index3, ap_ave4, ap_index4, img, title):

    plt.plot(ap_index1[10:],ap_ave1[10:],label='BiLSTM_512_0.15',color='r', marker='o', linestyle='-', linewidth=1.0)
    plt.plot(ap_index2[10:],ap_ave2[10:],label='BiLSTM_512_0.20',color='g', marker='*', linestyle='-', linewidth=1.0)
    plt.plot(ap_index3[10:],ap_ave3[10:],label='BiLSTM10_1024_0.2',color='b', marker='^', linestyle='-', linewidth=1.0)
    plt.plot(ap_index4[10:],ap_ave4[10:],label='BiLSTM10_1024_0.2_attention',color='m', marker='v', linestyle='-', linewidth=1.0)
    #plt.plot(ap_index,ap_thr,label='threshold',color='k', marker='v', linestyle='-', linewidth=1.0)

    plt.xlabel('Epochs')
    plt.ylabel('Average Precision')
    plt.title(title)
    plt.legend(loc=4)
    plt.savefig(img)

def ArgsHandle():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile1', help='data directory')
    parser.add_argument('logfile2', help='data directory')
    parser.add_argument('logfile3', help='data directory')
    parser.add_argument('logfile4', help='data directory')
    parser.add_argument('img', help='save as image')
    parser.add_argument('title', help='save as image')
    return parser.parse_args()
def main():
    args = ArgsHandle()
    ap_index1, ap_ave1 = Analysis_AP(args.logfile1)
    ap_index2, ap_ave2 = Analysis_AP(args.logfile2)
    ap_index3, ap_ave3 = Analysis_AP(args.logfile3)
    ap_index4, ap_ave4 = Analysis_AP(args.logfile4)
    plotimg(ap_ave1, ap_index1, ap_ave2, ap_index2, ap_ave3, ap_index3, ap_ave4, ap_index4, args.img,args.title)

if __name__ == '__main__':
    main()
