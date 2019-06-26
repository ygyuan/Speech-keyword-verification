#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data processing

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import numpy as np
import argparse
import os


def generate_matches_array(labels):
    N = len(labels)
    words=[]
    precision=[]
    for i in labels:
        j=i.split("_")
        words.append("_".join(j[:-4]))
        precision.append(j[-4])
    positive_matches = np.zeros(N * (N - 1) / 2, dtype=np.bool)
    negative_matches = np.zeros(N * (N - 1) / 2, dtype=np.bool)
    cur_matches_i = 0
    positive=0
    negative=0
    for i in range(N-2):
        for j in xrange(i+1,N-1):
            if words[i] == words[j]:
                if precision[i]=="1" and precision[i]== precision[j]:
                    positive_matches[cur_matches_i + j - (i+1) ]=True
                    positive+=1
                elif precision[i]!=precision[j]:
                    negative_matches[cur_matches_i + j - (i+1) ]=True
                    negative+=1
                else:
                    continue
        cur_matches_i += N-(i+1)
    print cur_matches_i, positive, np.sum(positive_matches==True), negative, np.sum(negative_matches==True)
    return (positive_matches, negative_matches)

def Padding(data,lengths):
    max_len=141 #np.max(lengths)
    Pdata = []
    for matrix in data:
        Pdata.append(np.pad(matrix, ((0, max_len - matrix.shape[0]), (0, 0)), mode='constant', constant_values=0))
    return np.asarray(Pdata)#.reshape(-1, 200, 40)#, np.array(lengths)

def GetNames(filepath):
    names = []
    train = np.load(filepath)
    for i in sorted(train):
        name = i[: i.find('_')]
        if name not in names:
            names.append(name)
    return names

def GetTestData(dict,flags):
    labels = []
    data = []
    lengths = []
    for i in sorted(dict.files):
        labels.append(i)
        data.append(dict[i])
        lengths.append(len(dict[i]))
    lengths = np.array(lengths)
    print np.histogram(lengths,bins=20)
    data = Padding(data,lengths)
    if flags:
        matches = generate_matches_array(labels)
        return data, lengths, matches, labels
    else:
        return data, lengths, labels

def BestAP(logfile):
    contents = open(logfile).read().split('\n')
    best_AP = 0
    idx_now = -6
    idx = -6
    for content in contents:
        if content.startswith('Dev'):
            idx_now += 5
            #t = float(content[content.find(': ') + 2: ])
            t = float(content[content.find(': ') + 2: ].split(" ")[0])
            if t > best_AP:
                best_AP = t
                idx = idx_now
    if idx<0:
        idx=0
    return best_AP, idx


def ModelClean(indices, path, output_name):
    for file in os.listdir(path):
        if file.startswith(output_name):
            print file
            version = file[file.find('-') + 1:]
            #if version.endswith('meta'):
            #    version = version[:-5]
            version = int(version[:version.find('.')])
            print version
            if version not in indices:
                os.remove(path + '/' + file)


def ArgsHandle():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='data directory')
    parser.add_argument('model_dir', help='model directory')
    parser.add_argument('output_dir',  help='output directory')
    parser.add_argument('-hs', default='512', help='LSTM hidden size')
    parser.add_argument('-m', default='0.4', help='fixed margin')
    parser.add_argument('-lr', default='0.0001', help='learning rate')
    parser.add_argument('-kp', default='0.6', help='keep probability')
    parser.add_argument('-bs', default='500', help='batch size')
    parser.add_argument('-epo', default='1000', help='epochs')
    parser.add_argument('-inits', default='0.05', help='LSTM cell initialization scale')
    parser.add_argument('-lastepoch', default='-1', help='train from which epoch, -1 means scratch')
    parser.add_argument('-th', default='11', help='threshold for sensitive margin')
    parser.add_argument('-sets', default='1', help='number of same word pairs')
    parser.add_argument('-n_same_pairs', default='500', help='number of same word pairs')
    parser.add_argument('-output_size', default='0', help='number of same word pairs')
    parser.add_argument('-gpu_device', default='0', help='number of same word pairs')
    return parser.parse_args()


def ModelName(args):
    return args.hs + '_' + args.m + '_' + args.lr + '_' + args.kp + '_' + args.bs + '_' + args.epo + '_' + args.inits +'_set' + args.sets + '_' + args.n_same_pairs  + '_' + args.output_size+ '-' + args.lastepoch


def OutputName(args):
    return args.hs + '_' + args.m + '_' + args.lr + '_' + args.kp + '_' + args.bs + '_' + args.epo + '_' + args.inits+ '_set'  + args.sets + '_' +  args.n_same_pairs + '_' + args.output_size


def main():
    args = ArgsHandle()
    print args
    pass

if __name__ == '__main__':
    main()
