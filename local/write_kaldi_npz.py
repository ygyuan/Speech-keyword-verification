#!/usr/bin/env python

"""
Write all matrices in a Kaldi archive to Numpy .npz format.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import argparse
import datetime
import numpy as np
import sys

import re


def read_kaldi_ark(ark_fn):
    """Read a Kaldi archive (in text format) and return it in a dict."""
    ark_dict = {}
    lines = open(ark_fn).readlines()
    for line in lines:
        line = line.strip(" \n")
        if line[-1] == "[":
            cur_id = line.split()[0]
            cur_mat = []
        elif "]" in line:
            line = line.strip("]")
            cur_mat.append([float(i) for i in line.split()])
            ark_dict[cur_id] = np.array(cur_mat)
        else:
            cur_mat.append([float(i) for i in line.split()])
    return ark_dict


def write_kaldi_ark(ark_dict, ark_fn):
    """
    Write the Kaldi archive dict `ark_dict` to `ark_fn` in text format.
    """
    np.set_printoptions(linewidth=np.nan, threshold=np.nan)
    f = open(ark_fn, "w")
    for cur_id in ark_dict:
        cur_mat = ark_dict[cur_id]
        f.write(cur_id + "  [\n")
        f.write(re.sub("(\]\])|(\])|(\ \[)|(\[\[)", " ", str(cur_mat)))
        f.write("]\n")
    f.close()
    np.set_printoptions(linewidth=75, threshold=1000)

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("kaldi_ark_fn", help="Kaldi archive file in text format")
    parser.add_argument("npz_fn", help="Numpy output file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Start time: " + str(datetime.datetime.now())

    print "Reading Kaldi archive"
    kaldi_ark = read_kaldi_ark(args.kaldi_ark_fn)
    print "Number of keys in archive:", len(kaldi_ark.keys())

    print "Writing feature vectors to file:", args.npz_fn
    np.savez(args.npz_fn, **kaldi_ark)

    print "End time: " + str(datetime.datetime.now())


if __name__ == "__main__":
    main()
