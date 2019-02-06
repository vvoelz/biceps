import os, sys, glob
import numpy as np
from scipy import loadtxt

import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import yaml
import numpy as np

#########################################
# Let's create our ensemble of structures

expdata_filename = 'cineromycinB_expdata_VAV.yaml'
energies_filename = 'cineromycinB_QMenergies.dat'
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()

lambdas =  np.arange(0.0, 1.2, 0.2)
#lambdas =  np.arange(0.0, 0.4, 0.2)

K = len(lambdas)
ensembles = []
samplers = []

for lam in lambdas:
    print 'lam =', lam
    # create an ensemble of structures with lambda-scaled free_energies
    ensemble = []
    for i in range(100):
        print '#### STRUCTURE %d ####'%i
        ensemble.append( Structure('cineromycinB_pdbs/%d.fixed.pdb'%i, lam*energies[i], expdata_filename, use_log_normal_distances=False) )
    ensembles.append( ensemble )
    samplers.append( PosteriorSampler(ensemble, dlogsigma_noe=np.log(1.02), sigma_noe_min=0.7, sigma_noe_max=0.71,
                                 dlogsigma_J=np.log(1.02), sigma_J_min=5.0, sigma_J_max=5.1,
                                 dloggamma=np.log(1.01), gamma_min=1.3, gamma_max=1.31,
                                 use_reference_prior=True) )


################

traj = []
for lam in lambdas:
    filename = 'traj_lambda_%1.1f.yaml'%lam
    print 'Loading', filename, '...'
    traj.append( yaml.load( file(filename, 'r') ) )


from pymbar import MBAR

# Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
#   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
u_kln = np.zeros( (K,K,len(traj[0]['trajectory'])) )
N_k = np.array( K*[len(traj[0]['trajectory'])] ) # N_k[k] will denote the number of correlated snapshots from state k

# Get snapshot energies rescored in the different ensembles
for k in range(K):
  for l in range(K):
    print 'samples from ensemble', k, 'evaluated in ensemble', l
    for i in range(N_k[k]):

        #if k == l:
        #    u_kln[k,l,i] = traj[k]['trajectory'][i][1] 
        #else:
        if (1):
            state, sigma_noe_index, sigma_J_index, gamma_index = traj[k]['trajectory'][i][3:]
            """ Reminder: ['step', 'E', 'accept', 'state', 'sigma_noe', 'sigma_J', 'gamma']"""
            sigma_noe = traj[k]['allowed_sigma_noe'][sigma_noe_index]
            sigma_J = traj[k]['allowed_sigma_J'][sigma_J_index]
            u_kln[k,l,i] = samplers[l].neglogP(state, sigma_noe, sigma_J, gamma_index)
 
print u_kln

# Initialize MBAR with reduced energies u_kln and number of uncorrelated configurations from each state N_k.
# 
# u_kln[k,l,n] is the reduced potential energy beta*U_l(x_kn), where U_l(x) is the potential energy function for state l,
# beta is the inverse temperature, and and x_kn denotes uncorrelated configuration n from state k.
#
# N_k[k] is the number of configurations from state k stored in u_knm
# 
# Note that this step may take some time, as the relative dimensionless free energies f_k are determined at this point.
mbar = MBAR(u_kln, N_k, verbose=True)
# OPTIONS
# maximum_iterations=10000, relative_tolerance=1e-07, verbose=False,
# initial_f_k=None, method='adaptive', use_optimized=None, newton_first_gamma=0.1, newton_self_consistent=2, maxrange=100000.0, initialize='zeros'

# Extract dimensionless free energy differences and their statistical uncertainties.
(Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
print 'Deltaf_ij', Deltaf_ij
print 'dDeltaf_ij', dDeltaf_ij
K = 2
beta = 1.0 # keep in units kT
print 'Unit-bearing (units kT) free energy difference between states 1 and K: %f +- %f' % ( (1./beta) * Deltaf_ij[0,K-1], (1./beta) * dDeltaf_ij[0,K-1])

# Compute the expectation of some observable A(x) at each state i, and associated uncertainty matrix.
# Here, A_kn[k,n] = A(x_{kn})
#(A_k, dA_k) = mbar.computeExpectations(A_kn)


sys.exit(1)

