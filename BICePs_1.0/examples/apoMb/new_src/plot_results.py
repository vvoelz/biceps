import sys, os, glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.append('./')

from Structure import *
from PosteriorSampler import *

import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR

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
legend_fontsize = 8

# Make a figure
#plt.figure( figsize=(6.5,6) )
plt.figure(figsize=(13,12))
# Make a subplot in the upper left
if (1):
    plt.subplot(4,2,1)
    # We assume: column 0 is lambda=0.0 and column K-1 is lambda=1.0
    plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')
    plt.hold(True)
    plt.plot([1e-6, 1], [1e-6, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-6, 1.)
    plt.ylim(1e-6, 1.)
    plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
    plt.ylabel('$p_i$ (MSM+exp)', fontsize=label_fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # label key states
    plt.hold(True)
    for i in range(len(pops1)):
        if (pops1[i] > 0.00):
            plt.text( pops0[i], pops1[i], str(i), color='g' )
    #for i in [87, 21, 79]:
    #    plt.text( pops0[i], pops1[i], str(i), color='r' )
    #for i in [38, 39, 45, 59, 65, 80, 85, 90, 92]:
    #    plt.text( pops0[i], pops1[i], str(i), color='g' )

    #plt.tight_layout()

if (0):
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

if (0):
    # plot histograms of sampled sigma_cs
    plt.subplot(3,2,2)
    plt.step(t0['allowed_sigma_cs_H'], t0['sampled_sigma_cs_H'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_cs_H'], t1['sampled_sigma_cs_H'], 'r-')
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,1.5)
    plt.xlabel("$\sigma_{cs_H}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs_H})$", fontsize=label_fontsize)
    plt.yticks([])

if (0):
    # plot histograms of sampled sigma_cs
    plt.subplot(3,2,3)
    plt.step(t0['allowed_sigma_cs_Ca'], t0['sampled_sigma_cs_Ca'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_cs_Ca'], t1['sampled_sigma_cs_Ca'], 'r-')
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,2.5)
    plt.xlabel("$\sigma_{cs_{C_a}}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs_{C_a}})$", fontsize=label_fontsize)
    plt.yticks([])

if (0):
    # plot histograms of sampled sigma_cs
    plt.subplot(3,2,4)
    plt.step(t0['allowed_sigma_cs_N'], t0['sampled_sigma_cs_N'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_cs_N'], t1['sampled_sigma_cs_N'], 'r-')
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,5.0)
    plt.xlabel("$\sigma_{cs_N}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs_N})$", fontsize=label_fontsize)
    plt.yticks([])

if (0):
    # plot histograms of sampled sigma_cs
    plt.subplot(2,2,2)
    plt.step(t0['allowed_sigma_cs'], t0['sampled_sigma_cs'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_cs'], t1['sampled_sigma_cs'], 'r-')
    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,1.5)
    plt.xlabel("$\sigma_{cs}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs})$", fontsize=label_fontsize)
    plt.yticks([])


if (0):
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

if (1):
    # plot histograms of sampled sigma_pf 
    plt.subplot(4,2,2)
    plt.step(t0['allowed_sigma_PF'], t0['sampled_sigma_PF'], 'b-')

    plt.hold(True)
    plt.step(t1['allowed_sigma_PF'], t1['sampled_sigma_PF'], 'r-')

    if (1):
        print '### sampled_sigma_PF'
        for i in range(len(t1['allowed_sigma_PF'])):
            print t1['allowed_sigma_PF'][i], t1['sampled_sigma_PF'][i]

    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
#    plt.xlim(0.0, 3.0)
    plt.xlabel("$\sigma_{PF}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{PF})$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])

if (0):
    # plot histograms of sampled alpha 
    plt.subplot(3,2,6)
    plt.step(t0['allowed_alpha'], t0['sampled_alpha'], 'b-')

    plt.hold(True)
    plt.step(t1['allowed_alpha'], t1['sampled_alpha'], 'r-')
  
    if (1):
        print '### sampled_alpha'
        for i in range(len(t1['allowed_alpha'])):
            print t1['allowed_alpha'][i], t1['sampled_alpha'][i]

    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlim(-2.0, 2.0)
    plt.xlabel(r"$\alpha$", fontsize=label_fontsize)
    plt.ylabel(r"$P(\alpha)$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])

if (1):
    # plot histograms of sampled_beta_c 
    plt.subplot(4,2,3)
    plt.step(t0['allowed_beta_c'], t0['sampled_beta_c'], 'b-')

    plt.hold(True)
    plt.step(t1['allowed_beta_c'], t1['sampled_beta_c'], 'r-')

    if (1):
        print '### sampled_beta_c'
        for i in range(len(t1['allowed_beta_c'])):
            print t1['allowed_beta_c'][i], t1['sampled_beta_c'][i]

    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(0.01, 0.1)
    plt.xlabel(r'$\beta_C$', fontsize=label_fontsize)
    plt.ylabel(r"$P(\beta_C)$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])

if (1):
    # plot histograms of sampled_beta_c 
    plt.subplot(4,2,4)
    plt.step(t0['allowed_beta_h'], t0['sampled_beta_h'], 'b-')

    plt.hold(True)
    plt.step(t1['allowed_beta_h'], t1['sampled_beta_h'], 'r-')

    if (1):
        print '### sampled_beta_h'
        for i in range(len(t1['allowed_beta_h'])):
            print t1['allowed_beta_h'][i], t1['sampled_beta_h'][i]

    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(-0.10, 2.10)
    plt.xlabel(r"$\beta_H$", fontsize=label_fontsize)
    plt.ylabel(r"$P(\beta_H)$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])

if (1):
    # plot histograms of sampled_beta_c 
    plt.subplot(4,2,5)
    plt.step(t0['allowed_beta_0'], t0['sampled_beta_0'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_beta_0'], t1['sampled_beta_0'], 'r-')
    if (1):
        print '### sampled_beta_0'
        for i in range(len(t1['allowed_beta_0'])):
            print t1['allowed_beta_0'][i], t1['sampled_beta_0'][i]
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(-3.2, 1.2)
    plt.xlabel(r"$\beta_0$", fontsize=label_fontsize)
    plt.ylabel(r"$P(\beta_0)$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])
if (1):
    # plot histograms of sampled_beta_c 
    plt.subplot(4,2,6)
    plt.step(t0['allowed_xcs'], t0['sampled_xcs'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_xcs'], t1['sampled_xcs'], 'r-')
    if (1):
        print '### sampled_xcs'
        for i in range(len(t1['allowed_xcs'])):
            print t1['allowed_xcs'][i], t1['sampled_xcs'][i]
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(4.0, 9.0)
    plt.xlabel("xc", fontsize=label_fontsize)
    plt.ylabel("P(xc)", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])
if (1):
    # plot histograms of sampled_beta_c 
    plt.subplot(4,2,7)
    plt.step(t0['allowed_xhs'], t0['sampled_xhs'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_xhs'], t1['sampled_xhs'], 'r-')
    if (1):
        print '### sampled_xhs'
        for i in range(len(t1['allowed_xhs'])):
            print t1['allowed_xhs'][i], t1['sampled_xhs'][i]
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(1.9, 2.8)
    plt.xlabel("xh", fontsize=label_fontsize)
    plt.ylabel("P(xh)", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])
if (0):
    # plot histograms of sampled_beta_c 
    plt.subplot(4,2,8)
    plt.step(t0['allowed_bs'], t0['sampled_bs'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_bs'], t1['sampled_bs'], 'r-')
    if (1):
        print '### sampled_bs'
        for i in range(len(t1['allowed_bs'])):
            print t1['allowed_bs'][i], t1['sampled_bs'][i]
    plt.legend(['exp', 'MSM+exp'], fontsize=legend_fontsize)
    plt.xlim(2.0, 6.0)
    plt.xlabel("b", fontsize=label_fontsize)
    plt.ylabel("P(b)", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])
plt.tight_layout()
plt.savefig('biceps3.pdf')
#plt.show()




