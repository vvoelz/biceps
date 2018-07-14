##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# Child class of PosteriorSampler for chemical shift NH.
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

class posterior_cs_H(object):
    def __init__(self,dlogsigma_cs_H=np.log(1.02),sigma_cs_H_min=0.05, sigma_cs_H_max=20.0):
        # pick initial values for sigma_cs_H (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_H = dlogsigma_cs_H  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_H_min = sigma_cs_H_min
        self.sigma_cs_H_max = sigma_cs_H_max
        self.allowed_sigma_cs_H = np.exp(np.arange(np.log(self.sigma_cs_H_min), np.log(self.sigma_cs_H_max), self.dlogsigma_cs_H))
        print 'self.allowed_sigma_cs_H', self.allowed_sigma_cs_H
        print 'len(self.allowed_sigma_cs_H) =', len(self.allowed_sigma_cs_H)
        self.sigma_cs_H_index = len(self.allowed_sigma_cs_H)/2   # pick an intermediate value to start with
        self.sigma_cs_H = self.allowed_sigma_cs_H[self.sigma_cs_H_index]


    def build_exp_ref_H(self):                          #GYH
        """Look at all the structures to find the average noe

        >>    beta_j = np.array(noe_distributions[j]).sum()/(len(noe_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials (H) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_H = len(ensemble[0].cs_H_restraints)
     #       print "ncs_H", ncs_H
            all_cs_H = []
            cs_H_distributions = [[] for j in range(ncs_H)]
            for s in ensemble:
                for j in range(len(s.cs_H_restraints)):
                    cs_H_distributions[j].append( s.cs_H_restraints[j].model_cs_H )
                    all_cs_H.append( s.cs_H_restraints[j].model_cs_H )

            # Find the MLE average (i.e. beta_j) for each noe
            betas_H = np.zeros(ncs_H)
            for j in range(ncs_H):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_H[j] =  np.array(cs_H_distributions[j]).sum()/(len(cs_H_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_H = betas_H
    #            print "s.betas_H", s.betas_H
                s.compute_neglog_exp_ref_H()

    def build_gau_ref_H(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference potentials (H) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            ncs_H = len(ensemble[0].cs_H_restraints)
            all_cs_H = []
            cs_H_distributions = [[] for j in range(ncs_H)]
            for s in ensemble:
                for j in range(len(s.cs_H_restraints)):
                    cs_H_distributions[j].append( s.cs_H_restraints[j].model_cs_H )
                    all_cs_H.append( s.cs_H_restraints[j].model_cs_H )

            # Find the MLE average (i.e. beta_j) for each noe
            ref_mean_H = np.zeros(ncs_H)
            ref_sigma_H = np.zeros(ncs_H)
            for j in range(ncs_H):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_H[j] =  np.array(cs_H_distributions[j]).mean()
                squared_diffs_H = [ (d - ref_mean_H[j])**2.0 for d in cs_H_distributions[j] ]
                ref_sigma_H[j] = np.sqrt( np.array(squared_diffs_H).sum() / (len(cs_H_distributions[j])+1.0))
            global_ref_sigma_H = ( np.array([ref_sigma_H[j]**-2.0 for j in range(ncs_H)]).mean() )**-0.5
            for j in range(ncs_H):
                ref_sigma_H[j] = global_ref_sigma_H
#               ref_sigma_H[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_H = ref_mean_H
                s.ref_sigma_H = ref_sigma_H
                s.compute_neglog_gau_ref_H()
	
