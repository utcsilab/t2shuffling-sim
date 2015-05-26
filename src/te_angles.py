#!/usr/bin/python

#import matplotlib.pyplot as plt
import sys
#import numpy as np


def parse_fliptable(logfile):
    f = open(logfile, 'r')
    angles = []

    for line in f.readlines():
        if 's_target' in line:
            angles.append(float(line.split('flip')[1].split('=')[1].split()[0]))
    f.close()
    return angles

if __name__ == '__main__':

    if len(sys.argv) < 2:
            print '%s <fliptable.log>' % sys.argv[0].split('/')[-1]
            print 'dump flip angles from fliptable log file'
            exit(-1)
    else:
            logfile = sys.argv[1]

    angles = parse_fliptable(logfile)
    for a in angles:
        print a
