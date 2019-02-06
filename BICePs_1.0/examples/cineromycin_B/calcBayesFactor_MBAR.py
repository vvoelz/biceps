import os, sys, glob
import numpy as np
from scipy import loadtxt

import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import yaml
import cPickle, pprint

#########################################
# Read in the pickled sampler objects

pkl_filename1, yaml_filename1 = 'sampler1.pkl', 'traj_exp.yaml'

pkl_filename2, yaml_filename2 = 'sampler2.pkl', 'traj_QMexp.yaml'
#pkl_filename2, yaml_filename2 = 'sampler3_noref.pkl', 'traj_exp_noref.yaml'
#pkl_filename2, yaml_filename2 = 'sampler4_noref.pkl', 'traj_QMexp_noref.yaml'


pkl_file = open(pkl_filename1, 'rb')
sampler1 = cPickle.load(pkl_file)
pprint.pprint(sampler1)

pkl_file = open(pkl_filename2, 'rb')
sampler2 = cPickle.load(pkl_file)
pprint.pprint(sampler2)

################

# Load in results from the exp-only sampling
print 'Loading %s...'%yaml_filename1
t = yaml.load( file(yaml_filename1, 'r') )

# Load in results from the QM+exp sampling
print 'Loading %s...'%yaml_filename2
t2 = yaml.load( file(yaml_filename2, 'r') )


from pymbar import MBAR

# Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
#   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
u_kln = np.zeros( (2,2,len(t['trajectory'])) )
N_k = np.array( [len(t['trajectory']), len(t['trajectory'])] ) # N_k[k] will denote the number of correlated snapshots from state k

# Get snapshot energies rescored in the different ensembles
"""['step', 'E', 'accept', 'state', 'sigma_noe', 'sigma_J', 'gamma']"""
print 'ensemble 1 samples:'
for i in range(len(t['trajectory'])):

    print 'step', t['trajectory'][i][0],
    print 'E1 evaluated in model1', t['trajectory'][i][1],
    u_kln[0,0,i] = t['trajectory'][i][1] 
    print 'E1 evaluated in model2',  
    state, sigma_noe_index, sigma_J_index, gamma_index = t['trajectory'][i][3:]
    sigma_noe = t['allowed_sigma_noe'][sigma_noe_index]
    sigma_J = t['allowed_sigma_J'][sigma_J_index]
    gamma   = t['allowed_gamma'][gamma_index]
    u_kln[0,1,i] = sampler2.neglogP(0, state, sigma_noe, sigma_J, gamma)
    print u_kln[0,1,i]

print 'ensemble 2 samples:'
for i in range(len(t2['trajectory'])):

    print 'step', t2['trajectory'][i][0], 
    print 'E2 evaluated in model2', t2['trajectory'][i][1],
    u_kln[1,1,i] = t2['trajectory'][i][1]
    print 'E2 evaluated in model1',
    state, sigma_noe_index, sigma_J_index, gamma_index = t2['trajectory'][i][3:]
    sigma_noe = t2['allowed_sigma_noe'][sigma_noe_index]
    sigma_J = t2['allowed_sigma_J'][sigma_J_index]
    gamma   = t2['allowed_gamma'][gamma_index]
    u_kln[1,0,i] = sampler1.neglogP(0, state, sigma_noe, sigma_J, gamma)
    print u_kln[1,0,i]


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
K = 2
beta = 1.0 # keep in units kT
print 'Unit-bearing (units kT) free energy difference f_1K = f_K - f_1: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])

# Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
# Here, A_kn[k,n] = A(x_{kn})
#(A_k, dA_k) = mbar.computeExpectations(A_kn)


