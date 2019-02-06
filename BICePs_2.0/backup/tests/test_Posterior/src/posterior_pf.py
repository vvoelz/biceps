##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# Child class of PosteriorSampler for protection factors.
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


class posterior_pf(object):
    def __init__(self, dlogsigma_pf=np.log(1.01), sigma_pf_min=0.05, sigma_pf_max=20.0):
        # pick initial values for sigma_pf (std of experimental uncertainty in NOE pf)
        self.dlogsigma_pf = dlogsigma_pf  # stepsize in log(sigma_pf) - i.e. grow/shrink multiplier
        self.sigma_pf_min = sigma_pf_min
        self.sigma_pf_max = sigma_pf_max
        self.allowed_sigma_pf = np.exp(np.arange(np.log(self.sigma_pf_min), np.log(self.sigma_pf_max), self.dlogsigma_pf))
        print 'self.allowed_sigma_pf', self.allowed_sigma_pf
        print 'len(self.allowed_sigma_pf) =', len(self.allowed_sigma_pf)
        self.sigma_pf_index = len(self.allowed_sigma_pf)/2    # pick an intermediate value to start with
        self.sigma_pf = self.allowed_sigma_pf[self.sigma_pf_index]

    def build_exp_ref_pf(self):        #GYH
        """Look at all the structures to find the average pf

        >>    beta_j = np.array(pf_distributions[j]).sum()/(len(pf_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference potentials (pf) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect pf distributions across all structures
            npf = len(ensemble[0].pf_restraints)
            all_pf = []
            pf_distributions = [[] for j in range(npf)]
            for s in ensemble:
                for j in range(len(s.pf_restraints)):
                    pf_distributions[j].append( s.pf_restraints[j].model_pf )
                    all_pf.append( s.pf_restraints[j].model_pf )

            # Find the MLE average (i.e. beta_j) for each pf
            betas_pf = np.zeros(npf)
            for j in range(npf):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_pf[j] =  np.array(pf_distributions[j]).sum()/(len(pf_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_pf = betas_pf
                s.compute_neglog_exp_ref_pf()

    def build_gau_ref_pf(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing gaussian reference potentials (pf) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect pf distributions across all structures
            npf = len(ensemble[0].pf_restraints)
#           print 'npf', npf
#           sys.exit()
            all_pf = []
            pf_distributions = [[] for j in range(npf)]
            for s in ensemble:
                for j in range(len(s.pf_restraints)):
                    pf_distributions[j].append( s.pf_restraints[j].model_pf )
                    all_pf.append( s.pf_restraints[j].model_pf )

            # Find the MLE average (i.e. beta_j) for each pf
            ref_mean_pf = np.zeros(npf)
            ref_sigma_pf = np.zeros(npf)
            for j in range(npf):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_pf[j] =  np.array(pf_distributions[j]).mean()
                squared_diffs_pf = [ (d - ref_mean_pf[j])**2.0 for d in pf_distributions[j] ]
                ref_sigma_pf[j] = np.sqrt( np.array(squared_diffs_pf).sum() / (len(pf_distributions[j])+1.0))
#            global_ref_sigma_pf = ( np.array([ref_sigma_pf[j]**-2.0 for j in range(npf)]).mean() )**-0.5
#           print 'global_ref_sigma_pf ', global_ref_sigma_pf
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_pf = ref_mean_pf
                s.ref_sigma_pf = ref_sigma_pf
                s.compute_neglog_gau_ref_pf()

