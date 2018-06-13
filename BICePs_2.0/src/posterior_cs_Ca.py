##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# Child class of PosteriorSampler for chemical shift Ca.
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

class posterior_cs_Ca(object):
    def __init__(self,dlogsigma_cs_Ca = np.log(1.02), sigma_cs_Ca_min = 0.05, sigma_cs_Ca_max = 20.0):

        # pick initial values for sigma_cs_Ca (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ca = dlogsigma_cs_Ca  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ca_min = sigma_cs_Ca_min
        self.sigma_cs_Ca_max = sigma_cs_Ca_max
        self.allowed_sigma_cs_Ca = np.exp(np.arange(np.log(self.sigma_cs_Ca_min), np.log(self.sigma_cs_Ca_max), self.dlogsigma_cs_Ca))
        print 'self.allowed_sigma_cs_Ca', self.allowed_sigma_cs_Ca
        print 'len(self.allowed_sigma_cs_Ca) =', len(self.allowed_sigma_cs_Ca)
        self.sigma_cs_Ca_index = len(self.allowed_sigma_cs_Ca)/2   # pick an intermediate value to start with
        self.sigma_cs_Ca = self.allowed_sigma_cs_Ca[self.sigma_cs_Ca_index]


    def build_exp_ref_Ca(self):                          #GYH
        """Look at all the structures to find the average noe

        >>    beta_j = np.array(noe_distributions[j]).sum()/(len(noe_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials (Ca) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_Ca = len(ensemble[0].cs_Ca_restraints)
            all_cs_Ca = []
            cs_Ca_distributions = [[] for j in range(ncs_Ca)]
            for s in ensemble:
                for j in range(len(s.cs_Ca_restraints)):
                    cs_Ca_distributions[j].append( s.cs_Ca_restraints[j].model_cs_Ca )
                    all_cs_Ca.append( s.cs_Ca_restraints[j].model_cs_Ca )

            # Find the MLE average (i.e. beta_j) for each noe
            betas_Ca = np.zeros(ncs_Ca)
            for j in range(ncs_Ca):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_Ca[j] =  np.array(cs_Ca_distributions[j]).sum()/(len(cs_Ca_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_Ca = betas_Ca
                s.compute_neglog_exp_ref_Ca()


    def build_gau_ref_Ca(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference potentials (Ca) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_Ca = len(ensemble[0].cs_Ca_restraints)
            all_cs_Ca = []
            cs_Ca_distributions = [[] for j in range(ncs_Ca)]
            for s in ensemble:
                for j in range(len(s.cs_Ca_restraints)):
                    cs_Ca_distributions[j].append( s.cs_Ca_restraints[j].model_cs_Ca )
                    all_cs_Ca.append( s.cs_Ca_restraints[j].model_cs_Ca )

            # Find the MLE average (i.e. beta_j) for each noe
            ref_mean_Ca = np.zeros(ncs_Ca)
            ref_sigma_Ca = np.zeros(ncs_Ca)
            for j in range(ncs_Ca):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_Ca[j] =  np.array(cs_Ca_distributions[j]).mean()
                squared_diffs_Ca = [ (d - ref_mean_Ca[j])**2.0 for d in cs_Ca_distributions[j] ]
                ref_sigma_Ca[j] = np.sqrt( np.array(squared_diffs_Ca).sum() / (len(cs_Ca_distributions[j])+1.0))
            global_ref_sigma_Ca = ( np.array([ref_sigma_Ca[j]**-2.0 for j in range(ncs_Ca)]).mean() )**-0.5
            for j in range(ncs_Ca):
                ref_sigma_Ca[j] = global_ref_sigma_Ca
#               ref_sigma_Ca[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_Ca = ref_mean_Ca
                s.ref_sigma_Ca = ref_sigma_Ca
                s.compute_neglog_gau_ref_Ca()



