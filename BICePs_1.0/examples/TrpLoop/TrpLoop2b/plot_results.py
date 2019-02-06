import sys, os, glob

sys.path.append('../../../src')

from Structure import *
from PosteriorSampler import *

import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

fontfamily={'family':'sans-serif','sans-serif':['Arial']}
plt.rc('font', **fontfamily)



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("resultdir", help="the path name of the result directory")
parser.add_argument("popsfile", help="the path name of the population *.dat file")
args = parser.parse_args()

print '=== Settings ==='
print 'resultdir', args.resultdir
print 'popsfile', args.popsfile

# Load in yaml trajectories
exp_files = glob.glob( os.path.join(args.resultdir,'traj_lambda*.yaml') )


traj = []
for filename in exp_files:
    print 'Loading %s ...'%filename
    traj.append( yaml.load( file(filename, 'r') ) )

# Load in cpickled sampler objects
sampler_files = glob.glob( os.path.join(args.resultdir,'sampler_lambda*.pkl') )
sampler = []
for pkl_filename in sampler_files:
    print 'Loading %s ...'%pkl_filename
    pkl_file = open(pkl_filename, 'rb')
    sampler.append( cPickle.load(pkl_file) )

# parse the lambda* filenames to get the full list of lambdas
nlambda = len(exp_files)
lam = [float( (s.split('lambda')[1]).replace('.yaml','') ) for s in exp_files ]
print 'lam =', lam

K = nlambda

# Load in precomputed P and dP from calc_MBAR_fromresults....py
P_dP = loadtxt(args.popsfile)

pops0, pops1   = P_dP[:,0], P_dP[:,K-1]
dpops0, dpops1 = P_dP[:,K], P_dP[:,2*K-1]

t0 = traj[0]
t1 = traj[K-1]

######## PLOTTING ########


# Figure Plot SETTINGS
label_fontsize = 12
legend_fontsize = 10

# Make a figure
plt.figure( figsize=(6.5,6) )

# Make a subplot in the upper left
if (1):
    plt.subplot(2,2,1)
    # We assume: column 0 is lambda=0.0 and column K-1 is lambda=1.0
    plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')
    plt.hold(True)
    plt.plot([1e-6, 1], [1e-6, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-6, 1.)
    plt.ylim(1e-6, 1.)
    plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
    plt.ylabel('$p_i$ (REMD+exp)', fontsize=label_fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # label key states
    plt.hold(True)
    for i in range(len(pops1)):
        if (i==0) or (pops1[i] > 0.05):
            plt.text( pops0[i], pops1[i], str(i), color='g' )
    #for i in [87, 21, 79]:
    #    plt.text( pops0[i], pops1[i], str(i), color='r' )
    #for i in [38, 39, 45, 59, 65, 80, 85, 90, 92]:
    #    plt.text( pops0[i], pops1[i], str(i), color='g' )

    #plt.tight_layout()

if (1):
    # Plot histograms of sampled sigmal
    plt.subplot(2,2,2)
    plt.step(t0['allowed_sigma_noe'], t0['sampled_sigma_noe'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_noe'], t1['sampled_sigma_noe'], 'r-')
    if (1):
        print '### sampled_sigma'
        for i in range(len(t1['allowed_sigma_noe'])):
            print t1['allowed_sigma_noe'][i], t1['sampled_sigma_noe'][i]

    plt.xlim(0,6)
    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlabel("$\sigma_d$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_d)$", fontsize=label_fontsize)
    plt.yticks([])

if (0):
    # plot histograms of sampled sigma_J
    plt.subplot(2,2,3)
    plt.step(t0['allowed_sigma_J'], t0['sampled_sigma_J'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_J'], t1['sampled_sigma_J'], 'r-')
    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlabel("$\sigma_J$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_J)$", fontsize=label_fontsize)
    plt.yticks([])

if (1):
    # plot histograms of sampled sigma_cs
    plt.subplot(2,2,3)
    plt.step(t0['allowed_sigma_cs'], t0['sampled_sigma_cs'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_cs'], t1['sampled_sigma_cs'], 'r-')
    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,1.0)
    plt.xlabel("$\sigma_{cs}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs})$", fontsize=label_fontsize)
    plt.yticks([])


if (1):
    # plot histograms of sampled gamma 
    plt.subplot(2,2,4)
    plt.step(t0['allowed_gamma'], t0['sampled_gamma'], 'b-')

    plt.hold(True)
    plt.step(t1['allowed_gamma'], t1['sampled_gamma'], 'r-')
   
    if (1):
        print '### sampled_gamma'
        for i in range(len(t1['allowed_gamma'])):
            print t1['allowed_gamma'][i], t1['sampled_gamma'][i]

    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0, 5.0)
    plt.xlabel("$\gamma'$", fontsize=label_fontsize)
    plt.ylabel("$P(\gamma')$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])

plt.tight_layout()
plt.savefig('biceps3.pdf')
plt.show()




