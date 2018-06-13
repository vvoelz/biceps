##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# Child class of PosteriorSampler for chemical shift N.
##############################################################################

##############################################################################
# Imports
##############################################################################

import os, sys, glob
import numpy as np
import mdtraj

##############################################################################
# Code
##############################################################################

class posterior_cs_N(object):
    def __init__(self,dlogsigma_cs_N=np.log(1.02),sigma_cs_N_min=0.05, sigma_cs_N_max=20.0,):
        # pick initial values for sigma_cs_N (std of experimental uncertainty in chemical shift)   #GYN
        self.dlogsigma_cs_N = dlogsigma_cs_N  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_N_min = sigma_cs_N_min
        self.sigma_cs_N_max = sigma_cs_N_max
        self.allowed_sigma_cs_N = np.exp(np.arange(np.log(self.sigma_cs_N_min), np.log(self.sigma_cs_N_max), self.dlogsigma_cs_N))
        print 'self.allowed_sigma_cs_N', self.allowed_sigma_cs_N
        print 'len(self.allowed_sigma_cs_N) =', len(self.allowed_sigma_cs_N)
        self.sigma_cs_N_index = len(self.allowed_sigma_cs_N)/2   # pick an intermediate value to start with
        self.sigma_cs_N = self.allowed_sigma_cs_N[self.sigma_cs_N_index]


    def build_exp_ref_N(self):                          #GYN
        """Look at all the structures to find the average noe

        >>    beta_j = np.array(noe_distributions[j]).sum()/(len(noe_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials (N) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_N = len(ensemble[0].cs_N_restraints)
     #       print "ncs_N", ncs_N
            all_cs_N = []
            cs_N_distributions = [[] for j in range(ncs_N)]
            for s in ensemble:
                for j in range(len(s.cs_N_restraints)):
                    cs_N_distributions[j].append( s.cs_N_restraints[j].model_cs_N )
                    all_cs_N.append( s.cs_N_restraints[j].model_cs_N )

            # Find the MLE average (i.e. beta_j) for each noe
            betas_N = np.zeros(ncs_N)
            for j in range(ncs_N):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_N[j] =  np.array(cs_N_distributions[j]).sum()/(len(cs_N_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_N = betas_N
    #            print "s.betas_N", s.betas_N
                s.compute_neglog_exp_ref_N()

    def build_gau_ref_N(self):        #GYN

        for k in range(self.nensembles):

            print 'Computing Gaussian reference potentials (N) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_N = len(ensemble[0].cs_N_restraints)
            all_cs_N = []
            cs_N_distributions = [[] for j in range(ncs_N)]
            for s in ensemble:
                for j in range(len(s.cs_N_restraints)):
                    cs_N_distributions[j].append( s.cs_N_restraints[j].model_cs_N )
                    all_cs_N.append( s.cs_N_restraints[j].model_cs_N )

            # Find the MLE average (i.e. beta_j) for each noe
            ref_mean_N = np.zeros(ncs_N)
            ref_sigma_N = np.zeros(ncs_N)
            for j in range(ncs_N):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_N[j] =  np.array(cs_N_distributions[j]).mean()
                squared_diffs_N = [ (d - ref_mean_N[j])**2.0 for d in cs_N_distributions[j] ]
                ref_sigma_N[j] = np.sqrt( np.array(squared_diffs_N).sum() / (len(cs_N_distributions[j])+1.0))
            global_ref_sigma_N = ( np.array([ref_sigma_N[j]**-2.0 for j in range(ncs_N)]).mean() )**-0.5
            for j in range(ncs_N):
                ref_sigma_N[j] = global_ref_sigma_N
#               ref_sigma_N[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_N = ref_mean_N
                s.ref_sigma_N = ref_sigma_N
                s.compute_neglog_gau_ref_N()
	
