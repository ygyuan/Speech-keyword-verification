#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate keyword lists.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import argparse
import sys

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("text", help="Numpy output file")
    parser.add_argument("keywordslist_old", help="Numpy output file")
    parser.add_argument("keywordslist_new", help="Numpy output file")
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
    for line in open(args.keywordslist_old):
        line2 = unicode(line, "utf-8").split("\n")[0].split(" ")
        keywordslist[line2[0].encode("utf-8")]=line
    print len(keywordslist), keywordslist.keys()[0]

    f=open(args.keywordslist_new,"w")
    for line in open(args.text):
        line = unicode(line, "utf-8").split("\n")[0].split(" ")
        word = line[1].encode("utf-8")
        if word in keywordslist.keys():
            f.write(keywordslist[word])
            del keywordslist[word]
    print "sucess"

if __name__ == "__main__":
    main()
