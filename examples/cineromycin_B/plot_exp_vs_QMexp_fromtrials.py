import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR


# Load in yaml trajectories

# Load in results from the exp-only sampling
#exp_files = glob.glob('results_trial*/traj_exp.yaml')
exp_files = ['results_trial1/traj_exp.yaml']
t_list = []
for filename in exp_files:
    print 'Loading %s ...'%filename
    t_list.append( yaml.load( file(filename, 'r') ) )

# Load in results from the QM+exp sampling
#QMexp_files = glob.glob('results_trial*/traj_QMexp.yaml')
QMexp_files = ['results_trial1/traj_QMexp.yaml']
t2_list = []
for filename in QMexp_files:
    print 'Loading %s ...'%filename
    t2_list.append( yaml.load( file(filename, 'r') ) )

t = t_list[0]
t2 = t2_list[0]

# Load in cpickled sampler objects
#sampler1_files = glob.glob('results_trial*/sampler1.pkl')
sampler1_files = ['results_trial1/sampler1.pkl']
sampler1_list = []
for pkl_filename in sampler1_files:
    print 'Loading %s ...'%pkl_filename
    pkl_file = open(pkl_filename, 'rb')
    sampler1_list.append( cPickle.load(pkl_file) )

#sampler2_files = glob.glob('results_trial*/sampler2.pkl')
sampler2_files = ['results_trial1/sampler2.pkl']
sampler2_list = []
for pkl_filename in sampler2_files:
    print 'Loading %s ...'%pkl_filename
    pkl_file = open(pkl_filename, 'rb')
    sampler2_list.append( cPickle.load(pkl_file) )

sampler1 = sampler1_list[0]
sampler2 = sampler2_list[0]


# Load in precomputed P and dP from calc_pops_Bayes....py
P_dP = loadtxt('populations_MBAR.dat')
pops, dpops = P_dP[:,0:2], P_dP[:,2:4]

######## PLOTTING ########


# Figure Plot SETTINGS
label_fontsize = 18
legend_fontsize = 16

# Make a figure
plt.figure()

# Make a subplot in the upper left
plt.subplot(2,2,1)
if (1):
        plt.errorbar( pops[:,0], pops[:,1], xerr=dpops[:,0], yerr=dpops[:,1], fmt='k.')
        plt.hold(True)
        #for i in np.arange(nstates):
        #    if f2[i] < 4:
        #        plt.text( f[i], f2[i], str(Ind[i]) )
        #    elif f[i] < 3:
        #        plt.text( f[i], f2[i], str(Ind[i]) )
        #    else:
        #        pass
        #plt.plot([0, 6], [0, 6], color='k', linestyle='-', linewidth=2)
        plt.xlim(1e-4, 1.)
        plt.ylim(1e-5, 1.)
        plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
        plt.ylabel('$p_i$ (QM+exp)', fontsize=label_fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()


if (1):
	# Plot histograms of sampled sigmal
        plt.subplot(2,2,2)
        plt.step(t['allowed_sigma_noe'], t['sampled_sigma_noe'], 'b-')
        plt.hold(True)
        plt.step(t2['allowed_sigma_noe'], t2['sampled_sigma_noe'], 'r-')
        plt.xlim(0,3)
        plt.legend(['exp', 'QM+exp'], fontsize=legend_fontsize)
        plt.xlabel("$\sigma_d$", fontsize=label_fontsize)
        plt.ylabel("$P(\sigma_d)$", fontsize=label_fontsize)
        plt.yticks([])

        # plot histograms of sampled sigma_J
        plt.subplot(2,2,3)
        plt.step(t['allowed_sigma_J'], t['sampled_sigma_J'], 'b-')
        plt.hold(True)
        plt.step(t2['allowed_sigma_J'], t2['sampled_sigma_J'], 'r-')
        plt.legend(['exp', 'QM+exp'], fontsize=legend_fontsize)
        plt.xlabel("$\sigma_J$", fontsize=label_fontsize)
        plt.ylabel("$P(\sigma_J)$", fontsize=label_fontsize)
        plt.yticks([])

        # plot histograms of sampled gamma 
        plt.subplot(2,2,4)
        plt.step(t['allowed_gamma'], t['sampled_gamma'], 'b-')
        plt.hold(True)
        plt.step(t2['allowed_gamma'], t2['sampled_gamma'], 'r-')
        plt.legend(['exp', 'QM+exp'], fontsize=legend_fontsize)
        plt.xlim(0.8, 2.1)
        plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
        plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
        plt.yticks([])


        plt.show()




