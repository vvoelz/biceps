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
exp_files = glob.glob('results_trial*/traj_exp.yaml')
#exp_files = ['results_trial1/traj_exp.yaml']
t_list = []
for filename in exp_files:
    print 'Loading %s ...'%filename
    t_list.append( yaml.load( file(filename, 'r') ) )

# Load in results from the QM+exp sampling
QMexp_files = glob.glob('results_trial*/traj_QMexp.yaml')
#QMexp_files = ['results_trial1/traj_QMexp.yaml']
t2_list = []
for filename in QMexp_files:
    print 'Loading %s ...'%filename
    t2_list.append( yaml.load( file(filename, 'r') ) )

t = t_list[0]
t2 = t2_list[0]

# Load in cpickled sampler objects
sampler1_files = glob.glob('results_trial*/sampler1.pkl')
#sampler1_files = ['results_trial1/sampler1.pkl']
sampler1_list = []
for pkl_filename in sampler1_files:
    print 'Loading %s ...'%pkl_filename
    pkl_file = open(pkl_filename, 'rb')
    sampler1_list.append( cPickle.load(pkl_file) )

sampler2_files = glob.glob('results_trial*/sampler2.pkl')
#sampler2_files = ['results_trial1/sampler2.pkl']
sampler2_list = []
for pkl_filename in sampler2_files:
    print 'Loading %s ...'%pkl_filename
    pkl_file = open(pkl_filename, 'rb')
    sampler2_list.append( cPickle.load(pkl_file) )

sampler1 = sampler1_list[0]
sampler2 = sampler2_list[0]

ntrials = len(exp_files)


#### MBAR for populations ####

# Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
#   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
nlambda = 2
K = nlambda
nsnaps = len(t['trajectory'])
u_kln = np.zeros( (K, K, nsnaps*ntrials) )
nstates = 100
states_kn = np.zeros( (K, nsnaps*ntrials) )
N_k = np.array( [nsnaps*ntrials, nsnaps*ntrials] ) # N_k[k] will denote the number of correlated snapshots from state k

# Get snapshot energies rescored in the different ensembles
"""['step', 'E', 'accept', 'state', 'sigma_noe', 'sigma_J', 'gamma']"""


for b in range(ntrials): 
  t = t_list[b]
  t2 = t2_list[b]
  print 'ensemble 1 samples:'
  for i in range(nsnaps):

    print 'step', t['trajectory'][i][0],
    print 'E1 evaluated in model1', t['trajectory'][i][1],
    u_kln[0,0,b*ntrials+i] = t['trajectory'][i][1] 
    print 'E1 evaluated in model2',  
    state, sigma_noe_index, sigma_J_index, gamma_index = t['trajectory'][i][3:]
    states_kn[0,b*ntrials+i] = state
    sigma_noe = t['allowed_sigma_noe'][sigma_noe_index]
    sigma_J = t['allowed_sigma_J'][sigma_J_index]
    gamma   = t['allowed_gamma'][gamma_index]
    u_kln[0,1,b*ntrials+i] = sampler2.neglogP(0, state, sigma_noe, sigma_J, gamma)
    print u_kln[0,1,b*ntrials+i]

  print 'ensemble 2 samples:'
  for i in range(nsnaps):

    print 'step', t2['trajectory'][i][0], 
    print 'E2 evaluated in model2', t2['trajectory'][i][1],
    u_kln[1,1,b*ntrials+i] = t2['trajectory'][i][1]
    print 'E2 evaluated in model1',
    state, sigma_noe_index, sigma_J_index, gamma_index = t2['trajectory'][i][3:]
    states_kn[1,b*ntrials+i] = state
    sigma_noe = t2['allowed_sigma_noe'][sigma_noe_index]
    sigma_J = t2['allowed_sigma_J'][sigma_J_index]
    gamma   = t2['allowed_gamma'][gamma_index]
    u_kln[1,0,b*ntrials+i] = sampler1.neglogP(0, state, sigma_noe, sigma_J, gamma)
    print u_kln[1,0,b*ntrials+i]


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
print 'Deltaf_ij', Deltaf_ij
print 'dDeltaf_ij', dDeltaf_ij
beta = 1.0 # keep in units kT
print 'Unit-bearing (units kT) free energy difference f_1K = f_K - f_1: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])

# Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
# Here, A_kn[k,n] = A(x_{kn})
#(A_k, dA_k) = mbar.computeExpectations(A_kn)
P_dP = np.zeros( (nstates, 2*K) )  # left columns are P, right columns are dP
if (1):
    print 'state\tP\tdP'
    for i in range(nstates):
        A_kn = np.where(states_kn==i,1,0)
        (p_i, dp_i) = mbar.computeExpectations(A_kn, uncertainty_method='approximate')
        P_dP[i,0:2] = p_i
        P_dP[i,2:4] = dp_i
        print i,
        for p in p_i: print p,
        for dp in dp_i: print dp,
        print

pops, dpops = P_dP[:,0:2], P_dP[:,2:4]

print 'Writing populations_MBAR.dat...'
savetxt('populations_MBAR.dat', P_dP)
print '...Done.'



