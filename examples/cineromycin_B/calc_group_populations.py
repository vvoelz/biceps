import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

from matplotlib import pyplot as plt
from scipy import loadtxt, savetxt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("popsfile", help="the filename containing state populations and their uncertainties")
parser.add_argument("-v", "--verbose", help="print verbosely", action="store_true")
args = parser.parse_args()

print '=== Settings ==='
print 'popsfile', args.popsfile
print 'verbose', args.verbose

# Load populations
P_dP = loadtxt(args.popsfile)
K = P_dP.shape[1]/2 
pops = P_dP[:,K-1]   # assume lambda=1.00 is the last column in each half
print 'pops', pops

names = ['Group A', 'Group B', 'Group C', 'Group D']
groups = [[38, 39, 65, 90], [45, 59], [80,85], [92]]

for i in range(len(groups)):
    group = groups[i]
    group_pop = 0.0
    for state in group:
        group_pop += pops[state]
        print state, pops[state]
    print 'population of', names[i], group_pop




