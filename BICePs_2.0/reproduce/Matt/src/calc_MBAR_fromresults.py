import sys, os, glob

sys.path.append('./')

from Structure import *
from PosteriorSampler import *

import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("resultdir", help="the name of the result directory")
parser.add_argument("bayesfactorfile", help="the filename to write free energies of each ensemble")
parser.add_argument("popsfile", help="the filename to write state populations and their uncertainties") 
args = parser.parse_args()

print '=== Settings ==='
print 'resultdir', args.resultdir

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


#### MBAR for populations ####

# Suppose the energies sampled from each simulation are u_kln, where u_kln[k,l,n] is the reduced potential energy
#   of snapshot n \in 1,...,N_k of simulation k \in 1,...,K evaluated at reduced potential for state l.
K = nlambda   # number of thermodynamic ensembles
# N_k[k] will denote the number of correlated snapshots from state k
N_k = np.array( [len(traj[i]['trajectory']) for i in range(nlambda)] )
nsnaps = N_k.max()
u_kln = np.zeros( (K, K, nsnaps) )
nstates = 25
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
      print 'E_%d evaluated in model_%d'%(k,l), u_kln[k,l,n]


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
    print 'state\tP\tdP'
    for i in range(nstates):
        A_kn = np.where(states_kn==i,1,0)
        (p_i, dp_i) = mbar.computeExpectations(A_kn, uncertainty_method='approximate')
        P_dP[i,0:K] = p_i
        P_dP[i,K:2*K] = dp_i
        print i,
        for p in p_i: print p,
        for dp in dp_i: print dp,
        print

pops, dpops = P_dP[:,0:K], P_dP[:,K:2*K]

print 'Writing %s...'%args.popsfile
savetxt(args.popsfile, P_dP)
print '...Done.'



