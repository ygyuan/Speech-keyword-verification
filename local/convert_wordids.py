#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert wordids.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import argparse
import numpy as np
import sys

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("keywordslist", help="Kaldi archive file in text format")
    parser.add_argument("npz", help="Numpy output file")
    parser.add_argument("text", help="Numpy output file")
    parser.add_argument("npz_new", help="Numpy output file")
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
        keywordslist[line[0].encode("utf-8")]="_".join(line[1:])
    print len(keywordslist), keywordslist.keys()[0]

    textlist={}
    dir_name=args.text.split("/")[-2].split("_hires")[0]
    print dir_name
    pos=0
    neg=0
    for line in open(args.text):
        #print unicode(line, "utf-8").split("\n")[0].split(" ")
        line = unicode(line, "utf-8").split("\n")[0].split(" ")[:3]
        uttid, word, precision = line
        word = word.encode("utf-8")
        if word not in keywordslist:
            print word, " not in text"
            continue
        precision=precision.split("\r")[0].encode("utf-8")
        #print uttid, word, precision
        if precision=="是" or precision=="正确":
            textlist[uttid]="_".join([keywordslist[word],"1",dir_name, uttid])
            pos+=1
        elif precision=="否" or precision=="错误": #or precision=="无唤醒词":
            textlist[uttid]="_".join([keywordslist[word],"0",dir_name, uttid])
            neg+=1
        else:
            continue
            #print line, uttid, word, precision, "; discard!"
    print len(textlist.keys()),textlist.keys()[0]
    print pos, neg

    npz=np.load(args.npz)
    npz_new={}
    for uttid in npz:
        if uttid not in textlist:
            continue
        ids=textlist[uttid]
        npz_new[ids]=npz[uttid]
    np.savez(args.npz_new, **npz_new)
    print "sucess on ", len(npz_new.keys()), "examples"
if __name__ == "__main__":
    main()
