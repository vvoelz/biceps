"""
I combined scripts of MBAR calculation and figure parts together so we only need to load the data once and save much time and memory usage. Now there will be more arguments for input and "datafile" and "picfile" are not required. By default they will be "False" and if there is no input files for "datafile", the scripts will only finish MBAR calculation and get the population and BICePs score files. If there are input files for "datafile" then the figure part will be initialized. Based on the extension of input file the scripts can figure out what plots they need to make (pop, sigma_noe, gamma, sigma_cs_H, etc.). Based on the number of subplots it needs to make, the number of column and rows will be automatically determined. For now, the column is fixed to be 2. Also there is a default name for the output figure. It will be over-written if the users have their own preference. They need to add that as the argument though.  --Yunhui Ge 05/2018
"""


import sys, os, glob

sys.path.append('../src')

from Structure import *
from PosteriorSampler import *

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("states", help="number of microstates")
parser.add_argument("--dataFiles","-f",required=False,help="Glob pattern for data")  # RMR
parser.add_argument("resultdir", help="the name of the result directory")
parser.add_argument("bayesfactorfile", help="the filename to write free energies of each ensemble")
parser.add_argument("popsfile", help="the filename to write state populations and their uncertainties") 
parser.add_argument("--picfile","-s",required=False,help="the path name of the pic *.pdf file")
args = parser.parse_args()
#print args.dataFiles
#sys.exit()
if args.dataFiles is not None:
    d_l=[]
    if ',' in args.dataFiles:
        print 'Sorting out the data...\n'
        dir_list = (args.dataFiles).split(',')
        data = [[],[],[],[],[],[]] # list for every extension
        # Sorting the data by extension into lists. Various directories is not an issue...
        for i in range(0,len(dir_list)):
            convert = lambda txt: int(txt) if txt.isdigit() else txt
            # This convert / sorted glob is a bit fishy... needs many tests
            for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
                if j.endswith('.noe'):
                    data[0].append(j)
                elif j.endswith('.J'):
                    data[1].append(j)
                elif j.endswith('.cs_H'):
                    data[2].append(j)
                elif j.endswith('.cs_Ha'):
                    data[3].append(j)
                elif j.endswith('.cs_N'):
                    data[4].append(j)
                elif j.endswith('.cs_Ca'):
                    data[5].append(j)
                else:
                    raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
        data = np.array(filter(None, data)) # removing any empty lists
        Data = np.stack(data, axis=-1)
        data = Data.tolist()
        #print data,'\n\n'
    else:
        print 'Sorting out the data...\n'
        data = sorted(glob.glob(args.dataFiles))
#    print data
#    for i in data[0]:
#    	print i.endswith('cs_H')
#    sys.exit()
    for i in data[0]:
        print i
        if i.endswith('.noe'):
            d_l.append('sigma_noe')
            d_l.append('gamma')
        elif i.endswith('.cs_H'):
            d_l.append('sigma_cs_H')
        elif i.endswith('.cs_Ha'):
            d_l.append('sigma_cs_Ha')
        elif i.endswith('.cs_N'):
            d_l.append('sigma_cs_N')
        elif i.endswith('.cs_Ca'):
            d_l.append('sigma_cs_Ca')
        elif i.endswith('.J'):
            d_l.append('sigma_J')
        else:
            raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha,cs_N,cs_Ca}")
         

#print '=== Settings ==='
print 'resultdir', args.resultdir
print 'nstates', args.states
# Load in yaml trajectories
exp_files = glob.glob( os.path.join(args.resultdir,'traj_lambda*.yaml') )
exp_files.sort()

traj = []
for filename in exp_files:
    print 'Loading %s ...'%filename
    traj.append( yaml.load( file(filename, 'r') ) )

# Load in cpickled sampler objects
sampler_files = glob.glob( os.path.join(args.resultdir,'sampler_lambda*.pkl') )
sampler_files.sort()
sampler = []
for pkl_filename in sampler_files:
    print 'Loading %s ...'%pkl_filename
    pkl_file = open(pkl_filename, 'rb')
    sampler.append( cPickle.load(pkl_file) )

# parse the lambda* filenames to get the full list of lambdas
nlambda = len(exp_files)
lam = [float( (s.split('lambda')[1]).replace('.yaml','') ) for s in exp_files ]
print 'lam =', lam


#### MBAR for populations ####

# Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
#   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
K = nlambda   # number of thermodynamic ensembles
# N_k[k] will denote the number of correlated snapshots from state k
N_k = np.array( [len(traj[i]['trajectory']) for i in range(nlambda)] )
nsnaps = N_k.max()
u_kln = np.zeros( (K, K, nsnaps) )
nstates = int(args.states)
print 'nstates', nstates
states_kn = np.zeros( (K, nsnaps) )

# Get snapshot energies rescored in the different ensembles
"""['step', 'E', 'accept', 'state', 'sigma_noe', 'sigma_J', 'sigma_cs', 'sigma_PF''gamma']
[int(step), float(self.E), int(accept), int(self.state), int(self.sigma_noe_index), int(self.sigma_J_index), int(self.sigma_cs_H_index), int(self.sigma_cs_Ha_index), int(self.sigma_cs_N_index), int(self.sigma_cs_Ca_index), int(self.sigma_PF_index), int(self.gamma_index)]				
"""			#GYH

for n in range(nsnaps):

  for k in range(K):
    for l in range(K):
      print 'step', traj[k]['trajectory'][n][0],
      if k==l:
          print 'E%d evaluated in model %d'%(k,k), traj[k]['trajectory'][n][1],
          u_kln[k,k,n] = traj[k]['trajectory'][n][1] 
      if (1):
          state, sigma_noe_index, sigma_J_index, sigma_cs_H_index, sigma_cs_Ha_index, sigma_cs_N_index, sigma_cs_Ca_index, sigma_PF_index, gamma_index, alpha_index = traj[k]['trajectory'][n][3:]
          print 'state, sigma_noe_index, sigma_J_index, sigma_cs_H_index, sigma_cs_Ha_index, sigma_cs_N_index, sigma_cs_Ca_index, sigma_PF_index, gamma_index, alpha_index', state, sigma_noe_index, sigma_J_index, sigma_cs_H_index, sigma_cs_Ha_index, sigma_cs_N_index, sigma_cs_Ca_index, sigma_PF_index, gamma_index, alpha_index	#GYH
          states_kn[k,n] = state
          sigma_noe = traj[k]['allowed_sigma_noe'][sigma_noe_index]
          sigma_J = traj[k]['allowed_sigma_J'][sigma_J_index]
          sigma_cs_H = traj[k]['allowed_sigma_cs_H'][sigma_cs_H_index]  #GYH
          sigma_cs_Ha = traj[k]['allowed_sigma_cs_Ha'][sigma_cs_Ha_index]  #GYH
          sigma_cs_N = traj[k]['allowed_sigma_cs_N'][sigma_cs_N_index]  #GYH
          sigma_cs_Ca = traj[k]['allowed_sigma_cs_Ca'][sigma_cs_Ca_index]  #GYH
	  sigma_PF = traj[k]['allowed_sigma_PF'][sigma_PF_index]  #GYH
          u_kln[k,l,n] = sampler[l].neglogP(0, state, sigma_noe, sigma_J, sigma_cs_H, sigma_cs_Ha, sigma_cs_N, sigma_cs_Ca, sigma_PF, gamma_index, alpha_index)	#GYH
#      print 'E_%d evaluated in model_%d'%(k,l), u_kln[k,l,n]


# Initialize MBAR with reduced energies u_kln and number of uncorrelated configurations from each state N_k.
# 
# u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where U_l(x) is the potential energy function for state l,
# beta is the inverse temperature, and and x_kn denotes uncorrelated configuration n from state k.
#
# N_k[k] is the number of configurations from state k stored in u_knm
# 
# Note that this step may take some time, as the relative dimensionless free energies f_k are determined at this point.
mbar = MBAR(u_kln, N_k)

# Extract dimensionless free energy differences and their statistical uncertainties.
(Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
#(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
#(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='approximate')
#print 'Deltaf_ij', Deltaf_ij
#print 'dDeltaf_ij', dDeltaf_ij
beta = 1.0 # keep in units kT
#print 'Unit-bearing (units kT) free energy difference f_1K = f_K - f_1: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])
f_df = np.zeros( (nlambda, 2) )  # first column is Deltaf_ij[0,:], second column is dDeltaf_ij[0,:]
f_df[:,0] = Deltaf_ij[0,:]
f_df[:,1] = dDeltaf_ij[0,:]
print 'Writing %s...'%args.bayesfactorfile
savetxt(args.bayesfactorfile, f_df)
print '...Done.'


# Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
# Here, A_kn[k,n] = A(x_{kn})
#(A_k, dA_k) = mbar.computeExpectations(A_kn)
P_dP = np.zeros( (nstates, 2*K) )  # left columns are P, right columns are dP
if (1):
#    print 'state\tP\tdP'
    for i in range(nstates):
        A_kn = np.where(states_kn==i,1,0)
        (p_i, dp_i) = mbar.computeExpectations(A_kn, uncertainty_method='approximate')
        P_dP[i,0:K] = p_i
        P_dP[i,K:2*K] = dp_i
#        print i,
        for p in p_i: print p,
        for dp in dp_i: print dp,
#        print

pops, dpops = P_dP[:,0:K], P_dP[:,K:2*K]

print 'Writing %s...'%args.popsfile
savetxt(args.popsfile, P_dP)
print '...Done.'


if args.dataFiles is not None:
    fontfamily={'family':'sans-serif','sans-serif':['Arial']}
    plt.rc('font', **fontfamily)
    if args.picfile == None:
        args.picfile = 'BICePs.pdf'

#print '=== Settings ==='
#print 'resultdir', args.resultdir
#print 'popsfile', args.popsfile
#print 'picfile', args.picfile

# Load in yaml trajectories
#exp_files = glob.glob( os.path.join(args.resultdir,'traj_lambda*.yaml') )
#exp_files.sort()

#traj = []
#for filename in exp_files:
#    print 'Loading %s ...'%filename
#    traj.append( yaml.load( file(filename, 'r') ) )

# Load in cpickled sampler objects
#sampler_files = glob.glob( os.path.join(args.resultdir,'sampler_lambda*.pkl') )
#sampler_files.sort()
#sampler = []
#for pkl_filename in sampler_files:
#    print 'Loading %s ...'%pkl_filename
#    pkl_file = open(pkl_filename, 'rb')
#    sampler.append( cPickle.load(pkl_file) )

# parse the lambda* filenames to get the full list of lambdas
#nlambda = len(exp_files)
#lam = [float( (s.split('lambda')[1]).replace('.yaml','') ) for s in exp_files ]
#print 'lam =', lam

#K = nlambda

# Load in precomputed P and dP from calc_MBAR_fromresults....py
#P_dP = loadtxt(args.popsfile)

    pops0, pops1   = P_dP[:,0], P_dP[:,K-1]
    dpops0, dpops1 = P_dP[:,K], P_dP[:,2*K-1]

    t0 = traj[0]
    t1 = traj[K-1]
######## PLOTTING ########


    # Figure Plot SETTINGS
    label_fontsize = 12
    legend_fontsize = 10

    # Make a figure
    plt.figure( figsize=(8,10) )
    if (len(d_l)+1)%2 != 0:
        c,r = 2, (len(d_l)+2)/2
    else:
        c,r = 2, (len(d_l)+1)/2
    # Make a subplot in the upper left
    plt.subplot(r,c,1)
    plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')
    plt.hold(True)
    plt.plot([1e-6, 1], [1e-6, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-6, 1.)
    plt.ylim(1e-6, 1.)
    plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
    plt.ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # label key states
    plt.hold(True)
    for i in range(len(pops1)):
        if (i==0) or (pops1[i] > 0.05):
            plt.text( pops0[i], pops1[i], str(i), color='g' )
    for k in range(len(d_l)):
        plt.subplot(r,c,k+2)
        plt.step(t0['allowed_'+d_l[k]], t0['sampled_'+d_l[k]], 'b-')
        plt.hold(True)
        plt.xlim(0,5)
        plt.step(t1['allowed_'+d_l[k]], t1['sampled_'+d_l[k]], 'r-')
        plt.legend(['exp', 'sim+exp'], fontsize=legend_fontsize)
	if d_l[k].find('cs') == -1:
            plt.xlabel("$\%s$"%d_l[k], fontsize=label_fontsize)
            plt.ylabel("$P(\%s)$"%d_l[k], fontsize=label_fontsize)
            plt.yticks([])
        else:
            plt.xlabel("$\sigma_{%s}$"%d_l[k][6:],fontsize=label_fontsize)
            plt.ylabel("$P(\sigma_{%s})$"%d_l[k][6:],fontsize=label_fontsize)
            plt.yticks([])

    plt.tight_layout()
    plt.savefig(args.picfile)

sys.exit()
if (1):
    plt.subplot(3,2,1)
    # We assume: column 0 is lambda=0.0 and column K-1 is lambda=1.0
    plt.errorbar( pops0, pops1, xerr=dpops0, yerr=dpops1, fmt='k.')
    plt.hold(True)
    plt.plot([1e-6, 1], [1e-6, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-6, 1.)
    plt.ylim(1e-6, 1.)
    plt.xlabel('$p_i$ (exp)', fontsize=label_fontsize)
    plt.ylabel('$p_i$ (sim+exp)', fontsize=label_fontsize)
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
    plt.subplot(3,2,2)
    plt.step(t0['allowed_sigma_noe'], t0['sampled_sigma_noe'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_noe'], t1['sampled_sigma_noe'], 'r-')
    if (1):
        print '### sampled_sigma'
        for i in range(len(t1['allowed_sigma_noe'])):
            print t1['allowed_sigma_noe'][i], t1['sampled_sigma_noe'][i]

    plt.xlim(0,6)
    plt.legend(['exp', 'sim+exp'], fontsize=legend_fontsize)
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
    plt.subplot(3,2,3)
    plt.step(t0['allowed_sigma_cs_H'], t0['sampled_sigma_cs_H'], 'b-')
    plt.hold(True)

    plt.step(t1['allowed_sigma_cs_H'], t1['sampled_sigma_cs_H'], 'r-')
    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,2.0)
    plt.xlabel("$\sigma_{cs_H}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs_H})$", fontsize=label_fontsize)
    plt.yticks([])

if (1):
    # plot histograms of sampled sigma_cs
    plt.subplot(3,2,4)
    plt.step(t0['allowed_sigma_cs_Ha'], t0['sampled_sigma_cs_Ha'], 'b-')
    plt.hold(True)
    plt.step(t1['allowed_sigma_cs_Ha'], t1['sampled_sigma_cs_Ha'], 'r-')
    plt.legend(['exp', 'REMD+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0,1.0)
    plt.xlabel("$\sigma_{cs_{H_a}}$", fontsize=label_fontsize)
    plt.ylabel("$P(\sigma_{cs_{H_a}})$", fontsize=label_fontsize)
    plt.yticks([])


if (1):
    # plot histograms of sampled gamma 
    plt.subplot(3,2,5)
    plt.step(t0['allowed_gamma'], t0['sampled_gamma'], 'b-')

    plt.hold(True)
    plt.step(t1['allowed_gamma'], t1['sampled_gamma'], 'r-')

    if (1):
        print '### sampled_gamma'
        for i in range(len(t1['allowed_gamma'])):
            print t1['allowed_gamma'][i], t1['sampled_gamma'][i]

    plt.legend(['exp', 'sim+exp'], fontsize=legend_fontsize)
    plt.xlim(0.0, 5.0)
    plt.xlabel("$\gamma'$", fontsize=label_fontsize)
    plt.ylabel("$P(\gamma')$", fontsize=label_fontsize)
    #plt.xlabel("$\gamma^{-1/6}$", fontsize=label_fontsize)
    #plt.ylabel("$P(\gamma^{-1/6})$", fontsize=label_fontsize)
    plt.yticks([])

plt.tight_layout()
plt.savefig(args.picfile)
#plt.show()


