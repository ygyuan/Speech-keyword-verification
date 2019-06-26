#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze statistics for a given word list.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2018
"""

import sys
import argparse
#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#
def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segments_utt", help="")
    parser.add_argument("wav_scp", help="")
    parser.add_argument("text", help="")
    if len(sys.argv)== 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    f_wav=open(args.wav_scp,"w")
    f_text=open(args.text,"w")
    #segments_utt={}
    count=0
    for line in open(args.segments_utt):
        line=line.split("\n")[0].split(" ")
        if len(line) != 5:
            print line
            quit()
        count +=1
        uttid=line[0].split("/")[-1].split(".")[0]+"-"+str(count)
        start=float(line[1])
        end=float(line[2])
        dur=float(end)-float(start)
        temp="{:.2f} {:.2f} |".format(start,dur)
        shellcmd=uttid+" wget -q -O - "+line[0]+" | /usr/bin/sox -t wav - -t wav - trim "+temp
        print shellcmd
        f_wav.write(shellcmd+"\n")
        confidence=float(line[4])
        temp="{:}".format(round(confidence*100))
        shellcmd=uttid+" "+line[3]+" 正确 "+temp
        print shellcmd
        f_text.write(shellcmd+"\n")

if __name__ == "__main__":
    main()
