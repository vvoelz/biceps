##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# Child class of PosteriorSampler for chemical shift HA.
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

class posterior_cs_Ha(object):
    def __init__(self,dlogsigma_cs_Ha = np.log(1.02), sigma_cs_Ha_min = 0.05, sigma_cs_Ha_max = 20.0):

        # pick initial values for sigma_cs_Ha (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ha = dlogsigma_cs_Ha  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ha_min = sigma_cs_Ha_min
        self.sigma_cs_Ha_max = sigma_cs_Ha_max
        self.allowed_sigma_cs_Ha = np.exp(np.arange(np.log(self.sigma_cs_Ha_min), np.log(self.sigma_cs_Ha_max), self.dlogsigma_cs_Ha))
        print 'self.allowed_sigma_cs_Ha', self.allowed_sigma_cs_Ha
        print 'len(self.allowed_sigma_cs_Ha) =', len(self.allowed_sigma_cs_Ha)
        self.sigma_cs_Ha_index = len(self.allowed_sigma_cs_Ha)/2   # pick an intermediate value to start with
        self.sigma_cs_Ha = self.allowed_sigma_cs_Ha[self.sigma_cs_Ha_index]


    def build_exp_ref_Ha(self):                          #GYH
        """Look at all the structures to find the average noe

        >>    beta_j = np.array(noe_distributions[j]).sum()/(len(noe_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials (Ha) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_Ha = len(ensemble[0].cs_Ha_restraints)
            all_cs_Ha = []
            cs_Ha_distributions = [[] for j in range(ncs_Ha)]
            for s in ensemble:
                for j in range(len(s.cs_Ha_restraints)):
                    cs_Ha_distributions[j].append( s.cs_Ha_restraints[j].model_cs_Ha )
                    all_cs_Ha.append( s.cs_Ha_restraints[j].model_cs_Ha )

            # Find the MLE average (i.e. beta_j) for each noe
            betas_Ha = np.zeros(ncs_Ha)
            for j in range(ncs_Ha):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_Ha[j] =  np.array(cs_Ha_distributions[j]).sum()/(len(cs_Ha_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_Ha = betas_Ha
                s.compute_neglog_exp_ref_Ha()


    def build_gau_ref_Ha(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference potentials (Ha) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_Ha = len(ensemble[0].cs_Ha_restraints)
            all_cs_Ha = []
            cs_Ha_distributions = [[] for j in range(ncs_Ha)]
            for s in ensemble:
                for j in range(len(s.cs_Ha_restraints)):
                    cs_Ha_distributions[j].append( s.cs_Ha_restraints[j].model_cs_Ha )
                    all_cs_Ha.append( s.cs_Ha_restraints[j].model_cs_Ha )

            # Find the MLE average (i.e. beta_j) for each noe
            ref_mean_Ha = np.zeros(ncs_Ha)
            ref_sigma_Ha = np.zeros(ncs_Ha)
            for j in range(ncs_Ha):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_Ha[j] =  np.array(cs_Ha_distributions[j]).mean()
                squared_diffs_Ha = [ (d - ref_mean_Ha[j])**2.0 for d in cs_Ha_distributions[j] ]
                ref_sigma_Ha[j] = np.sqrt( np.array(squared_diffs_Ha).sum() / (len(cs_Ha_distributions[j])+1.0))
            global_ref_sigma_Ha = ( np.array([ref_sigma_Ha[j]**-2.0 for j in range(ncs_Ha)]).mean() )**-0.5
            for j in range(ncs_Ha):
                ref_sigma_Ha[j] = global_ref_sigma_Ha
#               ref_sigma_Ha[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_Ha = ref_mean_Ha
                s.ref_sigma_Ha = ref_sigma_Ha
                s.compute_neglog_gau_ref_Ha()



