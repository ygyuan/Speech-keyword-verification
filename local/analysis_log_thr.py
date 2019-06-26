#!/usr/bin/env python

"""
Analysis log files.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def Analysis_AP(logfile, img, title):
    contents = open(logfile).read().split('\n')
    #best_AP = 0
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
    print max(ap_ave),ap_pos[ap_ave.index(max(ap_ave))],ap_neg[ap_ave.index(max(ap_ave))]

    plt.plot(ap_index,ap_ave,label='Average',color='r', marker='o', linestyle='-', linewidth=1.0)
    plt.plot(ap_index,ap_pos,label='positive',color='g', marker='*', linestyle='-', linewidth=1.0)
    plt.plot(ap_index,ap_neg,label='negative',color='b', marker='^', linestyle='-', linewidth=1.0)
    #plt.plot(ap_index,ap_thr,label='threshold',color='k', marker='v', linestyle='-', linewidth=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc=4)
    plt.savefig(img)

def ArgsHandle():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='data directory')
    parser.add_argument('img', help='save as image')
    parser.add_argument('title', help='save as image')
    return parser.parse_args()
def main():
    args = ArgsHandle()
    Analysis_AP(args.logfile, args.img, args.title)

if __name__ == '__main__':
    main()
