##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# Child class of PosteriorSampler for noe.
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


class posterior_noe(object):
    def __init__(self, dlogsigma_noe=np.log(1.01), sigma_noe_min=0.05, sigma_noe_max=20.0):
        # pick initial values for sigma_noe (std of experimental uncertainty in NOE noe)
        self.dlogsigma_noe = dlogsigma_noe  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_noe_min = sigma_noe_min
        self.sigma_noe_max = sigma_noe_max
        self.allowed_sigma_noe = np.exp(np.arange(np.log(self.sigma_noe_min), np.log(self.sigma_noe_max), self.dlogsigma_noe))
        print 'self.allowed_sigma_noe', self.allowed_sigma_noe
        print 'len(self.allowed_sigma_noe) =', len(self.allowed_sigma_noe)
        self.sigma_noe_index = len(self.allowed_sigma_noe)/2    # pick an intermediate value to start with
        self.sigma_noe = self.allowed_sigma_noe[self.sigma_noe_index]

    def build_exp_ref_noe(self):        #GYH
        """Look at all the structures to find the average noe

        >>    beta_j = np.array(noe_distributions[j]).sum()/(len(noe_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials (noe) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            nnoe = len(ensemble[0].noe_restraints)
            all_noe = []
            noe_distributions = [[] for j in range(nnoe)]
            for s in ensemble:
                for j in range(len(s.noe_restraints)):
                    noe_distributions[j].append( s.noe_restraints[j].model_noe )
                    all_noe.append( s.noe_restraints[j].model_noe )

            # Find the MLE average (i.e. beta_j) for each noe
            betas_noe = np.zeros(nnoe)
            for j in range(nnoe):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_noe[j] =  np.array(noe_distributions[j]).sum()/(len(noe_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_noe = betas_noe
                s.compute_neglog_exp_ref_noe()

    def build_gau_ref_noe(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing gaussian reference potentials (noe) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect noe distributions across all structures
            nnoe = len(ensemble[0].noe_restraints)
#           print 'nnoe', nnoe
#           sys.exit()
            all_noe = []
            noe_distributions = [[] for j in range(nnoe)]
            for s in ensemble:
                for j in range(len(s.noe_restraints)):
                    noe_distributions[j].append( s.noe_restraints[j].model_noe )
                    all_noe.append( s.noe_restraints[j].model_noe )

            # Find the MLE average (i.e. beta_j) for each noe
            ref_mean_noe = np.zeros(nnoe)
            ref_sigma_noe = np.zeros(nnoe)
            for j in range(nnoe):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_noe[j] =  np.array(noe_distributions[j]).mean()
                squared_diffs_noe = [ (d - ref_mean_noe[j])**2.0 for d in noe_distributions[j] ]
                ref_sigma_noe[j] = np.sqrt( np.array(squared_diffs_noe).sum() / (len(noe_distributions[j])+1.0))
#            global_ref_sigma_noe = ( np.array([ref_sigma_noe[j]**-2.0 for j in range(nnoe)]).mean() )**-0.5
#           print 'global_ref_sigma_noe ', global_ref_sigma_noe
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_noe = ref_mean_noe
                s.ref_sigma_noe = ref_sigma_noe
                s.compute_neglog_gau_ref_noe()

