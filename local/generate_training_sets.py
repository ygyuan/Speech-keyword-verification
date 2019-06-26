#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seprate train/dev/test set.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import argparse
#import datetime
import numpy as np
import sys
#import random
import os

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("keywordslist", help="Numpy output file")
    parser.add_argument("dirs", help="Numpy output file")
    parser.add_argument("npz_train", help="Numpy output file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    keywordslist={}
    for line in open(args.keywordslist):
        line = unicode(line, "utf-8").split("\n")[0].split(" ")
        keywordslist["_".join(line[1:])]=line[0].encode("utf-8")
    print len(keywordslist), keywordslist.keys()[0]

    featspathlist=[]
    for fpathe,dirs,fs in os.walk(args.dirs):
        for f in fs:
            if f=="feats_new.npz":
                featspathlist.append(os.path.join(fpathe,f))
    print featspathlist
    npz={}
    for x in featspathlist:
        npz1=np.load(x)
        for uttid in npz1:
            npz[uttid]=npz1[uttid]
    lists=sorted(npz.keys())
    print "all words: ",len(lists)

    wordclass={}
    for i in lists:
        word="_".join(i.split("_")[:-4])
        if word not in keywordslist:
            continue
        if word not in wordclass:
            wordclass[word]=[]
        wordclass[word].append(i)

    npz_train={}
    #npz_dev={}
    count=0
    count_pos=0
    count_neg=0
    for word in wordclass:
        word_pos=[i for i in wordclass[word] if i.startswith(word+"_1")]
        word_neg=[i for i in wordclass[word] if i.startswith(word+"_0")]
        #if len(word_pos)>12 and len(word_neg)>12:
        if len(word_pos)>10 and len(word_neg)>10:
            count+=1
            print count,keywordslist[word], word, len(wordclass[word]), len(word_pos), len(word_neg)
            #word_pos_dev=random.sample(word_pos, min(400,len(word_pos)))
            for i in word_pos:
                npz_train[i]=npz[i]
                count_pos+=1
            #word_neg_dev=random.sample(word_neg, min(400,len(word_neg)))
            for i in word_neg:
                npz_train[i]=npz[i]
                count_neg+=1
    print count,count_pos,count_neg,len(npz_train.keys())
    np.savez(args.npz_train, **npz_train)
if __name__ == "__main__":
    main()
