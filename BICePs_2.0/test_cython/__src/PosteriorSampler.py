##############################################################################
# Authors: Vincent Voelz, Yunhui Ge, Rob Raddi
# This file is used to do posterior sampling of BICePs calculation.
##############################################################################


##############################################################################
# Imports
##############################################################################

from __future__ import print_function
import os, sys, glob, copy
import numpy as np
from scipy  import loadtxt, savetxt
from matplotlib import pylab as plt
#import yaml
#import numpy as np
import yaml, io
import h5py
import pickle
import xml


from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_cs import *    # Class - creates Chemical shift restraint file
from prep_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from prep_J import *     # Class - creates J-coupling const. restraint file
from prep_pf import *     # Class - creates Protection factor restraint file   #GYH

import Restraint as R   # Import the Restraint Parent Class as R



# Class PosteriorSampler: {{{
class PosteriorSampler(object):
    """A class to perform posterior sampling of conformational populations"""

    # __init__:{{{
    def __init__(self, ensemble,
             dlogsigma_noe=np.log(1.01), sigma_noe_min=0.05, sigma_noe_max=20.0,
                 dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
                 dlogsigma_cs_H=np.log(1.02),sigma_cs_H_min=0.05, sigma_cs_H_max=20.0,
             dlogsigma_cs_Ha=np.log(1.02),sigma_cs_Ha_min=0.05, sigma_cs_Ha_max=20.0,
             dlogsigma_cs_N=np.log(1.02),sigma_cs_N_min=0.05, sigma_cs_N_max=20.0,
             dlogsigma_cs_Ca=np.log(1.02),sigma_cs_Ca_min=0.05, sigma_cs_Ca_max=20.0,
                 dlogsigma_pf=np.log(1.02),sigma_pf_min=0.05, sigma_pf_max=20.0,
                 use_reference_potential_noe = True,
             use_reference_potential_J = True,
                 use_reference_potential_H = True,
                 use_reference_potential_Ha = True,
                 use_reference_potential_N = True,
                 use_reference_potential_Ca = True,
                 use_reference_potential_pf = True,
                 use_gaussian_reference_potential_noe = False,
             use_gaussian_reference_potential_J = False,
                 use_gaussian_reference_potential_H = False,
                 use_gaussian_reference_potential_Ha = False,
                 use_gaussian_reference_potential_N = False,
                 use_gaussian_reference_potential_Ca = False,
                 use_gaussian_reference_potential_pf = False):


        # the ensemble is a list of Structure() objects
        self.ensembles = [ ensemble ]   # why a list inside of a list?
        self.nstates = len(ensemble)    # number of structures (elements)
        self.nensembles = len(self.ensembles)    # always be equal to 1
        self.ensemble_index = 0


        # pick initial values for sigma_noe (std of experimental uncertainty in NOE distances)
        self.dlogsigma_noe = dlogsigma_noe  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_noe_min = sigma_noe_min
        self.sigma_noe_max = sigma_noe_max
        self.allowed_sigma_noe = np.exp(np.arange(np.log(self.sigma_noe_min), np.log(self.sigma_noe_max), self.dlogsigma_noe))
        print( 'self.allowed_sigma_noe', self.allowed_sigma_noe)
        print( 'len(self.allowed_sigma_noe) =', len(self.allowed_sigma_noe))
        self.sigma_noe_index = len(self.allowed_sigma_noe)/2    # pick an intermediate value to start with
        self.sigma_noe = self.allowed_sigma_noe[self.sigma_noe_index]

        # pick initial values for sigma_J (std of experimental uncertainty in J-coupling constant)
        self.dlogsigma_J = dlogsigma_J  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_J_min = sigma_J_min
        self.sigma_J_max = sigma_J_max
        self.allowed_sigma_J = np.exp(np.arange(np.log(self.sigma_J_min), np.log(self.sigma_J_max), self.dlogsigma_J))
        print( 'self.allowed_sigma_J', self.allowed_sigma_J)
        print( 'len(self.allowed_sigma_J) =', len(self.allowed_sigma_J))

        self.sigma_J_index = len(self.allowed_sigma_J)/2   # pick an intermediate value to start with
        self.sigma_J = self.allowed_sigma_J[self.sigma_J_index]


        # pick initial values for sigma_cs_H (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_H = dlogsigma_cs_H  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_H_min = sigma_cs_H_min
        self.sigma_cs_H_max = sigma_cs_H_max
        self.allowed_sigma_cs_H = np.exp(np.arange(np.log(self.sigma_cs_H_min), np.log(self.sigma_cs_H_max), self.dlogsigma_cs_H))
        print( 'self.allowed_sigma_cs_H', self.allowed_sigma_cs_H)
        print( 'len(self.allowed_sigma_cs_H) =', len(self.allowed_sigma_cs_H))
        self.sigma_cs_H_index = len(self.allowed_sigma_cs_H)/2   # pick an intermediate value to start with
        self.sigma_cs_H = self.allowed_sigma_cs_H[self.sigma_cs_H_index]

        # pick initial values for sigma_cs_Ha (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ha = dlogsigma_cs_Ha  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ha_min = sigma_cs_Ha_min
        self.sigma_cs_Ha_max = sigma_cs_Ha_max
        self.allowed_sigma_cs_Ha = np.exp(np.arange(np.log(self.sigma_cs_Ha_min), np.log(self.sigma_cs_Ha_max), self.dlogsigma_cs_Ha))
        print( 'self.allowed_sigma_cs_Ha', self.allowed_sigma_cs_Ha)
        print( 'len(self.allowed_sigma_cs_Ha) =', len(self.allowed_sigma_cs_Ha))
        self.sigma_cs_Ha_index = len(self.allowed_sigma_cs_Ha)/2   # pick an intermediate value to start with
        self.sigma_cs_Ha = self.allowed_sigma_cs_Ha[self.sigma_cs_Ha_index]


        # pick initial values for sigma_cs_N (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_N = dlogsigma_cs_N  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_N_min = sigma_cs_N_min
        self.sigma_cs_N_max = sigma_cs_N_max
        self.allowed_sigma_cs_N = np.exp(np.arange(np.log(self.sigma_cs_N_min), np.log(self.sigma_cs_N_max), self.dlogsigma_cs_N))
        print( 'self.allowed_sigma_cs_N', self.allowed_sigma_cs_N)
        print( 'len(self.allowed_sigma_cs_N) =', len(self.allowed_sigma_cs_N))
        self.sigma_cs_N_index = len(self.allowed_sigma_cs_N)/2   # pick an intermediate value to start with
        self.sigma_cs_N = self.allowed_sigma_cs_N[self.sigma_cs_N_index]

        # pick initial values for sigma_cs_Ca (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ca = dlogsigma_cs_Ca  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ca_min = sigma_cs_Ca_min
        self.sigma_cs_Ca_max = sigma_cs_Ca_max
        self.allowed_sigma_cs_Ca = np.exp(np.arange(np.log(self.sigma_cs_Ca_min), np.log(self.sigma_cs_Ca_max), self.dlogsigma_cs_Ca))
        print( 'self.allowed_sigma_cs_Ca', self.allowed_sigma_cs_Ca)
        print( 'len(self.allowed_sigma_cs_Ca) =', len(self.allowed_sigma_cs_Ca))
        self.sigma_cs_Ca_index = len(self.allowed_sigma_cs_Ca)/2   # pick an intermediate value to start with
        self.sigma_cs_Ca = self.allowed_sigma_cs_Ca[self.sigma_cs_Ca_index]
        # pick initial values for sigma_pf (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_pf = dlogsigma_pf  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_pf_min = sigma_pf_min
        self.sigma_pf_max = sigma_pf_max
        self.allowed_sigma_pf = np.exp(np.arange(np.log(self.sigma_pf_min), np.log(self.sigma_pf_max), self.dlogsigma_pf))
#        print( 'self.allowed_sigma_pf', self.allowed_sigma_pf)
#        print( 'len(self.allowed_sigma_pf) =', len(self.allowed_sigma_pf))
        self.sigma_pf_index = len(self.allowed_sigma_pf)/2   # pick an intermediate value to start with
        self.sigma_pf = self.allowed_sigma_pf[self.sigma_pf_index]

        # pick initial values for gamma^(-1/6) (NOE scaling parameter)
        self.allowed_gamma = ensemble[0].allowed_gamma
#        print( 'self.allowed_gamma', self.allowed_gamma)
#        print( 'len(self.allowed_gamma) =', len(self.allowed_gamma))
        self.gamma_index = len(self.allowed_gamma)/2    # pick an intermediate value to start with
        self.gamma = self.allowed_gamma[self.gamma_index]


        # the initial state of the structural ensemble we're sampling from
        self.state = 0    # index in the ensemble
        self.E = 1.0e99   # initial energy
        self.accepted = 0
        self.total = 0
        self.write_traj = 1000  # step frequencies to write trajectory info
        # frequency of printing to the screen
        self.print_every = 1000 # debug
        # frequency of storing trajectory samples
        self.traj_every = 100
        # initialze restraint child class
        r_J = R.restraint_J()
        r_cs_Ca = R.restraint_cs_Ca()
        r_cs_H = R.restraint_cs_H()
        r_cs_N = R.restraint_cs_N()
        r_cs_Ha = R.restraint_cs_Ha()
        r_noe = R.restraint_noe()
#        self.allowed_sigma_noe = r_noe.allowed_sigma_noe
#        self.allowed_sigma_J = r_J.allowed_sigma_J
#        self.allowed_sigma_cs_H = r_cs_H.allowed_sigma_cs_H
#        self.allowed_sigma_cs_Ha = r_cs_Ha.allowed_sigma_cs_Ha
#        self.allowed_sigma_cs_N = r_cs_N.allowed_sigma_cs_N
#        self.allowed_sigma_cs_Ca = r_cs_Ca.allowed_sigma_cs_Ca
#        self.allowed_gamma = r_noe.allowed_gamma
#        self.sigma_noe = r_noe.sigma_noe
#        self.sigma_noe_index = r_noe.sigma_noe_index
#        self.sigma_J = r_J.sigma_J
#        self.sigma_J_index = r_J.sigma_J_index
#        self.sigma_cs_H = r_cs_H.sigma_cs_H
#        self.sigma_cs_H_index = r_cs_H.sigma_cs_H_index
#        self.sigma_cs_Ha = r_cs_Ha.sigma_cs_Ha
#        self.sigma_cs_Ha_index = r_cs_Ha.sigma_cs_Ha_index
#        self.sigma_cs_N = r_cs_N.sigma_cs_N
#        self.sigma_cs_N_index = r_cs_N.sigma_cs_N_index
#        self.sigma_cs_Ca = r_cs_Ca.sigma_cs_Ca
#        self.sigma_cs_Ca_index = r_cs_Ca.sigma_cs_Ca_index
#        self.gamma = r_noe.gamma
#        self.gamma_index = r_noe.gamma_index

        # keep track of what we sampled in a trajectory
        self.traj = PosteriorSamplingTrajectory(self.ensembles[0],
                self.allowed_sigma_noe,
                self.allowed_sigma_J,
                self.allowed_sigma_cs_H,
                self.allowed_sigma_cs_Ha,
                self.allowed_sigma_cs_N,
                self.allowed_sigma_cs_Ca,
                self.allowed_sigma_pf,
                self.allowed_gamma)

        # compile reference potential of distances from the uniform distribution of distances
        self.use_reference_potential_noe = use_reference_potential_noe
        self.use_reference_potential_J = use_reference_potential_J
        self.use_reference_potential_H = use_reference_potential_H
        self.use_reference_potential_Ha = use_reference_potential_Ha
        self.use_reference_potential_N = use_reference_potential_N
        self.use_reference_potential_Ca = use_reference_potential_Ca
        self.use_reference_potential_pf = use_reference_potential_pf
        self.use_gaussian_reference_potential_noe = use_gaussian_reference_potential_noe
        self.use_gaussian_reference_potential_J = use_gaussian_reference_potential_J
        self.use_gaussian_reference_potential_H = use_gaussian_reference_potential_H
        self.use_gaussian_reference_potential_Ha = use_gaussian_reference_potential_Ha
        self.use_gaussian_reference_potential_N = use_gaussian_reference_potential_N
        self.use_gaussian_reference_potential_Ca = use_gaussian_reference_potential_Ca
        self.use_gaussian_reference_potential_pf = use_gaussian_reference_potential_pf

        s = self.ensembles[0][0]
        if sum(s.sse_distances) != 0:
            if self.use_reference_potential_noe == True and self.use_gaussian_reference_potential_noe == True:
                self.build_gaussian_reference_potential_noe()
            if self.use_reference_potential_noe == True and self.use_gaussian_reference_potential_noe == False:
                self.build_reference_potential_noe()
        elif s.sse_dihedrals != 0:
            if self.use_reference_potential_J == True and self.use_gaussian_reference_potential_J == True:
                self.build_gaussian_reference_potential_J()
            if self.use_reference_potential_J == True and self.use_gaussian_reference_potential_J == False:
                self.build_reference_potential_J()
        elif s.sse_cs_H != 0:
            if self.use_reference_potential_H == True and self.use_gaussian_reference_potential_H == True:
                self.build_gaussian_reference_potential_H()
            if self.use_reference_potential_H == True and self.use_gaussian_reference_potential_H == False:
                self.build_reference_potential_H()
        elif s.sse_cs_Ha != 0:
            if self.use_reference_potential_Ha == True and self.use_gaussian_reference_potential_Ha == True:
                self.build_gaussian_reference_potential_Ha()
            if self.use_reference_potential_Ha == True and self.use_gaussian_reference_potential_Ha == False:
                self.build_reference_potential_Ha()
        elif s.sse_cs_N != 0:
            if self.use_reference_potential_N == True and self.use_gaussian_reference_potential_N == True:
                self.build_gaussian_reference_potential_N()
            if self.use_reference_potential_N == True and self.use_gaussian_reference_potential_N == False:
                self.build_reference_potential_N()
        elif s.sse_cs_H != 0:
            if self.use_reference_potential_Ca == True and self.use_gaussian_reference_potential_Ca == True:
                self.build_gaussian_reference_potential_Ca()
            if self.use_reference_potential_Ca == True and self.use_gaussian_reference_potential_Ca == False:
                self.build_reference_potential_Ca()
        elif s.sse_pf != 0:
            if self.use_reference_potential_pf == True and self.use_gaussian_reference_potential_pf == True:
                self.build_gaussian_reference_potential_pf()
            if self.use_reference_potential_pf == True and self.use_gaussian_reference_potential_pf == False:
                self.build_reference_potential_pf()



           # make a flag if using both exp and gaussian ref potentials for one restraint
#        if self.use_reference_potential_noe and self.use_gaussian_reference_potential_noe:
#                sys.exit("error: Cannot use different reference potentials for one exp restraint")
#        if self.use_reference_potential_H and self.use_gaussian_reference_potential_H:
#                sys.exit("error: Cannot use different reference potentials for one exp restraint")
#        if self.use_reference_potential_Ha and self.use_gaussian_reference_potential_Ha:
#                sys.exit("error: Cannot use different reference potentials for one exp restraint")
#        if self.use_reference_potential_N and self.use_gaussian_reference_potential_N:
#                sys.exit("error: Cannot use different reference potentials for one exp restraint")
#        if self.use_reference_potential_Ca and self.use_gaussian_reference_potential_Ca:
#                sys.exit("error: Cannot use different reference potentials for one exp restraint")
#       if self.use_reference_potential_pf and self.use_gaussian_reference_potential_pf:
#               sys.exit("error: Cannot use different reference potentials for one exp restraint")



        # VERY IMPORTANT: compute reference state self.logZ  for the free energies, so they are properly normalized #
        Z = 0.0
        for ensemble_index in range(self.nensembles):
            for s in self.ensembles[ensemble_index]:
                Z +=  np.exp(-s.free_energy)
        self.logZ = np.log(Z)

        # store this constant so we're not recalculating it all the time in neglogP
        self.ln2pi = np.log(2.0*np.pi)

    # Build Reference Prior NOE (Nuclear Overhauser effect):{{{
    def build_reference_potential_noe(self):    #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print( 'Computing reference potentials (noe) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ndistances = len(ensemble[0].distance_restraints)
            all_distances = []
            distance_distributions = [[] for j in range(ndistances)]
            for s in ensemble:
                for j in range(len(s.distance_restraints)):
                    distance_distributions[j].append( s.distance_restraints[j].model_distance )
                    all_distances.append( s.distance_restraints[j].model_distance )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_noe = np.zeros(ndistances)
            for j in range(ndistances):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_noe[j] =  np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_noe = betas_noe
                s.compute_neglog_reference_potentials_noe()
    #}}}

    # Build Gaussian Ref. Prior NOE:{{{
    def build_gaussian_reference_potential_noe(self):        #GYH

        for k in range(self.nensembles):

            print( 'Computing gaussian reference potentials (noe) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ndistances = len(ensemble[0].distance_restraints)
#           print( 'ndistances', ndistances)
#           sys.exit()
            all_distances = []
            distance_distributions = [[] for j in range(ndistances)]
            for s in ensemble:
                for j in range(len(s.distance_restraints)):
                    distance_distributions[j].append( s.distance_restraints[j].model_distance )
                    all_distances.append( s.distance_restraints[j].model_distance )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_noe = np.zeros(ndistances)
            ref_sigma_noe = np.zeros(ndistances)
            for j in range(ndistances):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_noe[j] =  np.array(distance_distributions[j]).mean()
                squared_diffs_noe = [ (d - ref_mean_noe[j])**2.0 for d in distance_distributions[j] ]
                ref_sigma_noe[j] = np.sqrt( np.array(squared_diffs_noe).sum() / (len(distance_distributions[j])+1.0))
#            global_ref_sigma_noe = ( np.array([ref_sigma_noe[j]**-2.0 for j in range(ndistances)]).mean() )**-0.5
#           print( 'global_ref_sigma_noe ', global_ref_sigma_noe)
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_noe = ref_mean_noe
                s.ref_sigma_noe = ref_sigma_noe
                s.compute_gaussian_neglog_reference_potentials_noe()
    #}}}

    # Build Ref. Prior H:{{{
    def build_reference_potential_H(self):                      #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print( 'Computing reference potentials (H) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_H = len(ensemble[0].cs_H_restraints)
     #       print( "ncs_H", ncs_H)
            all_cs_H = []
            cs_H_distributions = [[] for j in range(ncs_H)]
            for s in ensemble:
                for j in range(len(s.cs_H_restraints)):
                    cs_H_distributions[j].append( s.cs_H_restraints[j].model_cs_H )
                    all_cs_H.append( s.cs_H_restraints[j].model_cs_H )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_H = np.zeros(ncs_H)
            for j in range(ncs_H):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_H[j] =  np.array(cs_H_distributions[j]).sum()/(len(cs_H_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_H = betas_H
    #            print( "s.betas_H", s.betas_H)
                s.compute_neglog_reference_potentials_H()
    #}}}

    # Build Gaussain Ref H:{{{
    def build_gaussian_reference_potential_H(self):        #GYH

        for k in range(self.nensembles):

            print( 'Computing Gaussian reference potentials (H) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_H = len(ensemble[0].cs_H_restraints)
            all_cs_H = []
            cs_H_distributions = [[] for j in range(ncs_H)]
            for s in ensemble:
                for j in range(len(s.cs_H_restraints)):
                    cs_H_distributions[j].append( s.cs_H_restraints[j].model_cs_H )
                    all_cs_H.append( s.cs_H_restraints[j].model_cs_H )

            # Find the MLE average (i.e. beta_j) for each distance
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
#ref_sigma_H[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_H = ref_mean_H
                s.ref_sigma_H = ref_sigma_H
                s.compute_gaussian_neglog_reference_potentials_H()
    #}}}

    # Build Ref Prior Ha:{{{
    def build_reference_potential_Ha(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print( 'Computing reference potentials (Ha) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_Ha = len(ensemble[0].cs_Ha_restraints)
            all_cs_Ha = []
            cs_Ha_distributions = [[] for j in range(ncs_Ha)]
            for s in ensemble:
                for j in range(len(s.cs_Ha_restraints)):
                    cs_Ha_distributions[j].append( s.cs_Ha_restraints[j].model_cs_Ha )
                    all_cs_Ha.append( s.cs_Ha_restraints[j].model_cs_Ha )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_Ha = np.zeros(ncs_Ha)
            for j in range(ncs_Ha):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_Ha[j] =  np.array(cs_Ha_distributions[j]).sum()/(len(cs_Ha_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_Ha = betas_Ha
                s.compute_neglog_reference_potentials_Ha()
    #}}}

    # Build Gaussian Ref Prior Ha:{{{
    def build_gaussian_reference_potential_Ha(self):        #GYH

        for k in range(self.nensembles):

            print( 'Computing Gaussian reference potentials (Ha) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_Ha = len(ensemble[0].cs_Ha_restraints)
            all_cs_Ha = []
            cs_Ha_distributions = [[] for j in range(ncs_Ha)]
            for s in ensemble:
                for j in range(len(s.cs_Ha_restraints)):
                    cs_Ha_distributions[j].append( s.cs_Ha_restraints[j].model_cs_Ha )
                    all_cs_Ha.append( s.cs_Ha_restraints[j].model_cs_Ha )

            # Find the MLE average (i.e. beta_j) for each distance
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
#ref_sigma_Ha[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_Ha = ref_mean_Ha
                s.ref_sigma_Ha = ref_sigma_Ha
                s.compute_gaussian_neglog_reference_potentials_Ha()
    #}}}

    # Build Ref Prior N:{{{
    def build_reference_potential_N(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print( 'Computing reference potentials (N) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_N = len(ensemble[0].cs_N_restraints)
            all_cs_N = []
            cs_N_distributions = [[] for j in range(ncs_N)]
            for s in ensemble:
                for j in range(len(s.cs_N_restraints)):
                    cs_N_distributions[j].append( s.cs_N_restraints[j].model_cs_N )
                    all_cs_N.append( s.cs_N_restraints[j].model_cs_N )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_N = np.zeros(ncs_N)
            for j in range(ncs_N):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_N[j] =  np.array(cs_N_distributions[j]).sum()/(len(cs_N_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_N = betas_N
                s.compute_neglog_reference_potentials_N()

    #:}}}

    # Build Gaussian Ref Prior N:{{{
    def build_gaussian_reference_potential_N(self):        #GYH

        for k in range(self.nensembles):

            print( 'Computing Gaussian reference potentials (N) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_N = len(ensemble[0].cs_N_restraints)
            all_cs_N = []
            cs_N_distributions = [[] for j in range(ncs_N)]
            for s in ensemble:
                for j in range(len(s.cs_N_restraints)):
                    cs_N_distributions[j].append( s.cs_N_restraints[j].model_cs_N )
                    all_cs_N.append( s.cs_N_restraints[j].model_cs_N )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_N = np.zeros(ncs_N)
            ref_sigma_N = np.zeros(ncs_N)
            for j in range(ncs_N):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_N[j] =  np.array(cs_N_distributions[j]).mean()
                squared_diffs_N = [ (d - ref_mean_N[j])**2.0 for d in cs_N_distributions[j] ]
#                ref_sigma_N[j] = np.sqrt( np.array(squared_diffs_N).sum() / (len(cs_N_distributions[j])+1.0))
#            global_ref_sigma_N = ( np.array([ref_sigma_N[j]**-2.0 for j in range(ncs_N)]).mean() )**-0.5
            for j in range(ncs_N):
#                ref_sigma_N[j] = global_ref_sigma_N
                ref_sigma_N[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_N = ref_mean_N
                s.ref_sigma_N = ref_sigma_N
                s.compute_gaussian_neglog_reference_potentials_N()

    #:}}}

    # Build Ref Prior Ca:{{{
    def build_reference_potential_Ca(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print( 'Computing reference potentials (Ca) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_Ca = len(ensemble[0].cs_Ca_restraints)
            all_cs_Ca = []
            cs_Ca_distributions = [[] for j in range(ncs_Ca)]
            for s in ensemble:
                for j in range(len(s.cs_Ca_restraints)):
                    cs_Ca_distributions[j].append( s.cs_Ca_restraints[j].model_cs_Ca )
                    all_cs_Ca.append( s.cs_Ca_restraints[j].model_cs_Ca )
            # Find the MLE average (i.e. beta_j) for each distance
            betas_Ca = np.zeros(ncs_Ca)
            for j in range(ncs_Ca):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_Ca[j] =  np.array(cs_Ca_distributions[j]).sum()/(len(cs_Ca_distributions[j])+1.0)
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_Ca = betas_Ca
                s.compute_neglog_reference_potentials_Ca()
    #:}}}

    # Build Gaussian Ref Prior Ca:{{{
    def build_gaussian_reference_potential_Ca(self):        #GYH

        for k in range(self.nensembles):

            print( 'Computing Gaussian reference potentials (Ca) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ncs_Ca = len(ensemble[0].cs_Ca_restraints)
            all_cs_Ca = []
            cs_Ca_distributions = [[] for j in range(ncs_Ca)]
            for s in ensemble:
                for j in range(len(s.cs_Ca_restraints)):
                    cs_Ca_distributions[j].append( s.cs_Ca_restraints[j].model_cs_Ca )
                    all_cs_Ca.append( s.cs_Ca_restraints[j].model_cs_Ca )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_Ca = np.zeros(ncs_Ca)
            ref_sigma_Ca = np.zeros(ncs_Ca)
            for j in range(ncs_Ca):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_Ca[j] =  np.array(cs_Ca_distributions[j]).mean()
                squared_diffs_Ca = [ (d - ref_mean_Ca[j])**2.0 for d in cs_Ca_distributions[j] ]
#                ref_sigma_Ca[j] = np.sqrt( np.array(squared_diffs_Ca).sum() / (len(cs_Ca_distributions[j])+1.0))
#            global_ref_sigma_Ca = ( np.array([ref_sigma_Ca[j]**-2.0 for j in range(ncs_Ca)]).mean() )**-0.5
            for j in range(ncs_Ca):
#                ref_sigma_Ca[j] = global_ref_sigma_Ca
                ref_sigma_Ca[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_Ca = ref_mean_Ca
                s.ref_sigma_Ca = ref_sigma_Ca
                s.compute_gaussian_neglog_reference_potentials_Ca()

    #:}}}

    # Build Ref Prior pf:{{{
    def build_reference_potential_pf(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference potential for each structures"""

        for k in range(self.nensembles):

            print( 'Computing reference potentials (pf) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            npf = len(ensemble[0].pf_restraints)
            all_pf = []
            pf_distributions = [[] for j in range(npf)]
            for s in ensemble:
                for j in range(len(s.pf_restraints)):
                    pf_distributions[j].append( s.pf_restraints[j].model_pf )
                    all_pf.append( s.pf_restraints[j].model_pf )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_pf = np.zeros(npf)
            for j in range(npf):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_pf[j] =  np.array(pf_distributions[j]).sum()/(len(pf_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.betas_pf = betas_pf
                s.compute_neglog_reference_potentials_pf()

    #:}}}

    # Build Gaussian Ref Prior pf:{{{
    def build_gaussian_reference_potential_pf(self):        #GYH

        for k in range(self.nensembles):

            print( 'Computing Gaussian reference potentials (pf) for ensemble', k, 'of', self.nensembles, '...')
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            npf = len(ensemble[0].pf_restraints)
            all_pf = []
            pf_distributions = [[] for j in range(npf)]
            for s in ensemble:
                for j in range(len(s.pf_restraints)):
                    pf_distributions[j].append( s.pf_restraints[j].model_pf )
                    all_pf.append( s.pf_restraints[j].model_pf )
#           print( 'all_pf', all_pf)
            print( 'len(all_pf)', len(all_pf))

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_pf = np.zeros(npf)
            ref_sigma_pf = np.zeros(npf)
#           squared_diffs_pf = []
            for j in range(npf):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_pf[j] =  np.array(pf_distributions[j]).mean()
#                squared_diffs_pf.append( [ (d - ref_mean_pf[j])**2.0 for d in pf_distributions[j] ])
                squared_diffs_pf=( [ (d - ref_mean_pf[j])**2.0 for d in pf_distributions[j] ])
#           ref_sigma_pf[j] = np.sqrt( np.array(squared_diffs_pf).sum() / (len(pf_distributions[j])+1.0))
#            global_ref_sigma_pf = ( np.array([ref_sigma_pf[j]**-2.0 for j in range(npf)]).mean() )**-0.5

#            global_ref_sigma_pf = np.array(all_pf).std()
#            print( 'global_ref_sigma_pf', global_ref_sigma_pf)
#            sys.exit(1)

#np.sqrt( np.array(squared_diffs_pf).sum() / (len(squared_diffs_pf) + npf))
            for j in range(npf):
#                ref_sigma_pf[j] = global_ref_sigma_pf #np.sqrt( np.array(squared_diffs_pf).sum() / (len(pf_distributions[j])+1.0))
                ref_sigma_pf[j] = 20.0
#           ref_sigma_pf[j] = 1.84394160179     #np.sqrt( np.array(squared_diffs_pf).sum() / (len(pf_distributions[j])+1.0))
            # store the beta information in each structure and compute/store the -log P_potential
            for s in ensemble:
                s.ref_mean_pf = ref_mean_pf
                s.ref_sigma_pf =  ref_sigma_pf
                s.compute_gaussian_neglog_reference_potentials_pf()

    #:}}}

    # -log(P):{{{
    def neglogP(self, new_ensemble_index, new_state, new_sigma_noe, new_sigma_J, new_sigma_cs_H, new_sigma_cs_Ha, new_sigma_cs_N, new_sigma_cs_Ca, new_sigma_pf, new_gamma_index, verbose=True):        #GYH
        """Return -ln P of the current configuration."""

        # The current structure being sampled
        s = self.ensembles[new_ensemble_index][new_state]

        print( 's = ',s)
        # model terms
        result = s.free_energy  + self.logZ
        print( 'Result =',result)
        # distance terms
        #result += (Nj+1.0)*np.log(self.sigma_noe)
        #if s.sse_distances is not None:        # trying to fix a future warning:"comparison to `None` will result in an elementwise object comparison in the future."
        if sum(s.sse_distances) != 0:
            result += (s.Ndof_distances)*np.log(new_sigma_noe)  # for use with log-spaced sigma values
            result += s.sse_distances[new_gamma_index] / (2.0*new_sigma_noe**2.0)
            result += (s.Ndof_distances)/2.0*self.ln2pi  # for normalization
            if self.use_reference_potential_noe == True and self.use_gaussian_reference_potential_noe == True:
                result -= s.sum_gaussian_neglog_reference_potentials_noe
            if self.use_reference_potential_noe == True and self.use_gaussian_reference_potential_noe == False:
                result -= s.sum_neglog_reference_potentials_noe

        # dihedral terms
        if s.sse_dihedrals != 0:
                result += (s.Ndof_dihedrals)*np.log(new_sigma_J) # for use with log-spaced sigma values
                result += s.sse_dihedrals / (2.0*new_sigma_J**2.0)
                result += (s.Ndof_dihedrals)/2.0*self.ln2pi  # for normalization
                if self.use_reference_potential_J == True and self.use_gaussian_reference_potential_J == True:
                    result -= s.sum_gaussian_neglog_reference_potentials_J
                if self.use_reference_potential_J == True and self.use_gaussian_reference_potential_J == False:
                    result -= s.sum_neglog_reference_potentials_J

        # cs terms                           # GYH
        if s.sse_cs_H != 0:
                result += (s.Ndof_cs_H)*np.log(new_sigma_cs_H)
                result += s.sse_cs_H / (2.0*new_sigma_cs_H**2.0)
                result += (s.Ndof_cs_H)/2.0*self.ln2pi
                if self.use_reference_potential_H == True and self.use_gaussian_reference_potential_H == True:
                    result -= s.sum_gaussian_neglog_reference_potentials_H
                if self.use_reference_potential_H == True and self.use_gaussian_reference_potential_H == False:
                    result -= s.sum_neglog_reference_potentials_H

        if s.sse_cs_Ha != 0:
                result += (s.Ndof_cs_Ha)*np.log(new_sigma_cs_Ha)
                result += s.sse_cs_Ha / (2.0*new_sigma_cs_Ha**2.0)
                result += (s.Ndof_cs_Ha)/2.0*self.ln2pi
                if self.use_reference_potential_Ha == True and self.use_gaussian_reference_potential_Ha == True:
                    result -= s.sum_gaussian_neglog_reference_potentials_Ha
                if self.use_reference_potential_Ha == True and self.use_gaussian_reference_potential_Ha == False:
                    result -= s.sum_neglog_reference_potentials_Ha

        if s.sse_cs_N != 0:
                result += (s.Ndof_cs_N)*np.log(new_sigma_cs_N)
                result += s.sse_cs_N / (2.0*new_sigma_cs_N**2.0)
                result += (s.Ndof_cs_N)/2.0*self.ln2pi
                if self.use_reference_potential_N == True and self.use_gaussian_reference_potential_N == True:
                    result -= s.sum_gaussian_neglog_reference_potentials_N
                if self.use_reference_potential_N == True and self.use_gaussian_reference_potential_N == False:
                    result -= s.sum_neglog_reference_potentials_N


        if s.sse_cs_Ca != 0:
                result += (s.Ndof_cs_Ca)*np.log(new_sigma_cs_Ca)
                result += s.sse_cs_Ca / (2.0*new_sigma_cs_Ca**2.0)
                result += (s.Ndof_cs_Ca)/2.0*self.ln2pi
                if self.use_reference_potential_Ca == True and self.use_gaussian_reference_potential_Ca == True:
                    result -= s.sum_gaussian_neglog_reference_potentials_Ca
                if self.use_reference_potential_Ca == True and self.use_gaussian_reference_potential_Ca == False:
                    result -= s.sum_neglog_reference_potentials_Ca


        # pf terms                           # GYH
        if s.sse_pf != 0:
                result += (s.Ndof_pf)*np.log(new_sigma_pf)
                result += s.sse_pf / (2.0*new_sigma_pf**2.0)
                result += (s.Ndof_pf)/2.0*self.ln2pi
                if self.use_reference_potential_pf == True and self.use_gaussian_reference_potential_pf == True:
                    result -= s.sum_gaussian_neglog_reference_potentials_pf
                if self.use_reference_potential_pf == True and self.use_gaussian_reference_potential_pf == False:
                    result -= s.sum_neglog_reference_potentials_pf

        if verbose:
            print( 'state, f_sim', new_state, s.free_energy,)
            #print( 's.sse_distances[', new_gamma_index, ']', s.sse_distances[new_gamma_index], 's.Ndof_distances', s.Ndof_distances)
            #print( 's.sse_dihedrals', s.sse_dihedrals, 's.Ndof_dihedrals', s.Ndof_dihedrals)
            print( 's.sse_cs_H', s.sse_cs_H, 's.Ndof_cs_H', s.Ndof_cs_H) # GYH
            print( 's.sse_cs_Ha', s.sse_cs_Ha, 's.Ndof_cs_Ha', s.Ndof_cs_Ha) # GYH
            print( 's.sse_cs_N', s.sse_cs_N, 's.Ndof_cs_N', s.Ndof_cs_N) # GYH
            print( 's.sse_cs_Ca', s.sse_cs_Ca, 's.Ndof_cs_Ca', s.Ndof_cs_Ca) # GYH
            print( 's.sse_pf', s.sse_pf, 's.Ndof_pf', s.Ndof_pf)
            print( 's.sum_neglog_reference_potentials_noe', s.sum_neglog_reference_potentials_noe, 's.sum_neglog_reference_potentials_H', s.sum_neglog_reference_potentials_H, 's.sum_neglog_reference_potentials_Ha',s.sum_neglog_reference_potentials_Ha, 's.sum_neglog_reference_potentials_N', s.sum_neglog_reference_potentials_N, 's.sum_neglog_reference_potentials_Ca', s.sum_neglog_reference_potentials_Ca, 's.sum_neglog_reference_potentials_pf', s.sum_neglog_reference_potentials_pf    )#GYH
            print( 's.sum_gaussian_neglog_reference_potentials_noe', s.sum_gaussian_neglog_reference_potentials_noe, 's.sum_gaussian_neglog_reference_potentials_H', s.sum_gaussian_neglog_reference_potentials_H, 's.sum_gaussian_neglog_reference_potentials_Ha', s.sum_gaussian_neglog_reference_potentials_Ha, 's.sum_gaussian_neglog_reference_potentials_N', s.sum_gaussian_neglog_reference_potentials_N, 's.sum_gaussian_neglog_reference_potentials_Ca', s.sum_gaussian_neglog_reference_potentials_Ca, 's.sum_gaussian_neglog_reference_potentials_pf', s.sum_gaussian_neglog_reference_potentials_pf       )#GYH
        return result
    # }}}

## Compute -log( reference potentials (ALL Restraints) ):{{{
#    def compute_neglog_reference_potentials_noe(self):        #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print( 'self.betas', self.betas)
#
#        self.neglog_reference_potentials_noe = np.zeros(self.ndistances)
#        self.sum_neglog_reference_potentials_noe = 0.
#        for j in range(self.ndistances):
#            self.neglog_reference_potentials_noe[j] = np.log(self.betas_noe[j]) + self.distance_restraints[j].model_distance/self.betas_noe[j]
#            self.sum_neglog_reference_potentials_noe  += self.distance_restraints[j].weight * self.neglog_reference_potentials_noe[j]
#
#    def compute_gaussian_neglog_reference_potentials_noe(self):        #GYH
#       """An alternative option for reference potential based on Gaussian distribution"""
#       self.gaussian_neglog_reference_potentials_noe = np.zeros(self.ndistances)
#       self.sum_gaussian_neglog_reference_potentials_noe = 0.
#       for j in range(self.ndistances):
#           self.gaussian_neglog_reference_potentials_noe[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_noe[j]) + (self.distance_restraints[j].model_distance - self.ref_mean_noe[j])**2.0/(2*(self.ref_sigma_noe[j]**2.0))
#           self.sum_gaussian_neglog_reference_potentials_noe += self.distance_restraints[j].weight * self.gaussian_neglog_reference_potentials_noe[j]
#
#    def compute_neglog_reference_potentials_H(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print( 'self.betas', self.betas)
#
#        self.neglog_reference_potentials_H = np.zeros(self.ncs_H)
#        self.sum_neglog_reference_potentials_H = 0.
#        for j in range(self.ncs_H):
#            self.neglog_reference_potentials_H[j] = np.log(self.betas_H[j]) + self.cs_H_restraints[j].model_cs_H/self.betas_H[j]
#            self.sum_neglog_reference_potentials_H  += self.cs_H_restraints[j].weight * self.neglog_reference_potentials_H[j]
#
#    def compute_gaussian_neglog_reference_potentials_H(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_potentials_H = np.zeros(self.ncs_H)
#        self.sum_gaussian_neglog_reference_potentials_H = 0.
#        for j in range(self.ncs_H):
#            self.gaussian_neglog_reference_potentials_H[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_H[j]) + (self.cs_H_restraints[j].model_cs_H - self.ref_mean_H[j])**2.0/(2*self.ref_sigma_H[j]**2.0)
#            self.sum_gaussian_neglog_reference_potentials_H += self.cs_H_restraints[j].weight * self.gaussian_neglog_reference_potentials_H[j]
#
#
#    def compute_neglog_reference_potentials_Ha(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print( 'self.betas', self.betas)
#
#        self.neglog_reference_potentials_Ha = np.zeros(self.ncs_Ha)
#        self.sum_neglog_reference_potentials_Ha = 0.
#        for j in range(self.ncs_Ha):
#            self.neglog_reference_potentials_Ha[j] = np.log(self.betas_Ha[j]) + self.cs_Ha_restraints[j].model_cs_Ha/self.betas_Ha[j]
#            self.sum_neglog_reference_potentials_Ha  += self.cs_Ha_restraints[j].weight * self.neglog_reference_potentials_Ha[j]
#
#    def compute_gaussian_neglog_reference_potentials_Ha(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_potentials_Ha = np.zeros(self.ncs_Ha)
#        self.sum_gaussian_neglog_reference_potentials_Ha = 0.
#        for j in range(self.ncs_Ha):
#            self.gaussian_neglog_reference_potentials_Ha[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ha[j]) + (self.cs_Ha_restraints[j].model_cs_Ha - self.ref_mean_Ha[j])**2.0/(2*self.ref_sigma_Ha[j]**2.0)
#            self.sum_gaussian_neglog_reference_potentials_Ha += self.cs_Ha_restraints[j].weight * self.gaussian_neglog_reference_potentials_Ha[j]
#
#    def compute_neglog_reference_potentials_N(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print( 'self.betas', self.betas)
#
#        self.neglog_reference_potentials_N = np.zeros(self.ncs_N)
#        self.sum_neglog_reference_potentials_N = 0.
#        for j in range(self.ncs_N):
#            self.neglog_reference_potentials_N[j] = np.log(self.betas_N[j]) + self.cs_N_restraints[j].model_cs_N/self.betas_N[j]
#            self.sum_neglog_reference_potentials_N  += self.cs_N_restraints[j].weight * self.neglog_reference_potentials_N[j]
#
#    def compute_gaussian_neglog_reference_potentials_N(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_potentials_N = np.zeros(self.ncs_N)
#        self.sum_gaussian_neglog_reference_potentials_N = 0.
#        for j in range(self.ncs_N):
#            self.gaussian_neglog_reference_potentials_N[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_N[j]) + (self.cs_N_restraints[j].model_cs_N - self.ref_mean_N[j])**2.0/(2*self.ref_sigma_N[j]**2.0)
#            self.sum_gaussian_neglog_reference_potentials_N += self.cs_N_restraints[j].weight * self.gaussian_neglog_reference_potentials_N[j]
#
#
#    def compute_neglog_reference_potentials_Ca(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print( 'self.betas', self.betas)
#
#        self.neglog_reference_potentials_Ca = np.zeros(self.ncs_Ca)
#        self.sum_neglog_reference_potentials_Ca = 0.
#        for j in range(self.ncs_Ca):
#            self.neglog_reference_potentials_Ca[j] = np.log(self.betas_Ca[j]) + self.cs_Ca_restraints[j].model_cs_Ca/self.betas_Ca[j]
#            self.sum_neglog_reference_potentials_Ca  += self.cs_Ca_restraints[j].weight * self.neglog_reference_potentials_Ca[j]
#
#    def compute_gaussian_neglog_reference_potentials_Ca(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_potentials_Ca = np.zeros(self.ncs_Ca)
#        self.sum_gaussian_neglog_reference_potentials_Ca = 0.
#        for j in range(self.ncs_Ca):
#            self.gaussian_neglog_reference_potentials_Ca[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ca[j]) + (self.cs_Ca_restraints[j].model_cs_Ca - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
#            self.sum_gaussian_neglog_reference_potentials_Ca += self.cs_Ca_restraints[j].weight * self.gaussian_neglog_reference_potentials_Ca[j]
#
#
#    def compute_neglog_reference_potentials_pf(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print( 'self.betas', self.betas)
#
#        self.neglog_reference_potentials_pf= np.zeros(self.npf)
#        self.sum_neglog_reference_potentials_pf = 0.
#        for j in range(self.npf):
#            self.neglog_reference_potentials_pf[j] = np.log(self.betas_pf[j]) + self.pf_restraints[j].model_pf/self.betas_pf[j]
#            self.sum_neglog_reference_potentials_pf  += self.pf_restraints[j].weight * self.neglog_reference_potentials_pf[j]
#
#
#    def compute_gaussian_neglog_reference_potentials_pf(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_potentials_pf = np.zeros(self.npf)
#        self.sum_gaussian_neglog_reference_potentials_pf = 0.
#        for j in range(self.npf):
##          print( j, 'self.ref_sigma_pf[j]', self.ref_sigma_pf[j], 'self.ref_mean_pf[j]', self.ref_mean_pf[j])
#            self.gaussian_neglog_reference_potentials_pf[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_pf[j]) + (self.pf_restraints[j].model_pf - self.ref_mean_pf[j])**2.0/(2*self.ref_sigma_pf[j]**2.0)
#            self.sum_gaussian_neglog_reference_potentials_pf += self.pf_restraints[j].weight * self.gaussian_neglog_reference_potentials_pf[j]
#    # }}}
#
    # Sample:{{{
    def sample(self, nsteps):
        "Perform nsteps of posterior sampling."

        for step in range(nsteps):

            new_sigma_noe = self.sigma_noe
            new_sigma_noe_index = self.sigma_noe_index
            new_sigma_J = self.sigma_J
            new_sigma_J_index = self.sigma_J_index
            new_sigma_cs_H = self.sigma_cs_H                #GYH
            new_sigma_cs_H_index = self.sigma_cs_H_index    #GYH
            new_sigma_cs_Ha = self.sigma_cs_Ha                #GYH
            new_sigma_cs_Ha_index = self.sigma_cs_Ha_index    #GYH
            new_sigma_cs_N = self.sigma_cs_N                #GYH
            new_sigma_cs_N_index = self.sigma_cs_N_index    #GYH
            new_sigma_cs_Ca = self.sigma_cs_Ca                #GYH
            new_sigma_cs_Ca_index = self.sigma_cs_Ca_index    #GYH
            new_sigma_pf = self.sigma_pf                #GYH
            new_sigma_pf_index = self.sigma_pf_index    #GYH
            new_gamma = self.gamma
            new_gamma_index = self.gamma_index

            new_state = self.state
            new_ensemble_index = self.ensemble_index

            if np.random.random() < 0.16:
                # take a step in array of allowed sigma_distance
                new_sigma_noe_index += (np.random.randint(3)-1)
                new_sigma_noe_index = new_sigma_noe_index%(len(self.allowed_sigma_noe)) # don't go out of bounds
                new_sigma_noe = self.allowed_sigma_noe[new_sigma_noe_index]

            elif np.random.random() < 0.32:
                # take a step in array of allowed sigma_J
                new_sigma_J_index += (np.random.randint(3)-1)
                new_sigma_J_index = new_sigma_J_index%(len(self.allowed_sigma_J)) # don't go out of bounds
                new_sigma_J = self.allowed_sigma_J[new_sigma_J_index]

            elif np.random.random() < 0.48 :
                # take a step in array of allowed sigma_cs
                new_sigma_cs_H_index += (np.random.randint(3)-1)
                new_sigma_cs_H_index = new_sigma_cs_H_index%(len(self.allowed_sigma_cs_H)) # don't go out of bounds
                new_sigma_cs_H = self.allowed_sigma_cs_H[new_sigma_cs_H_index]
                new_sigma_cs_Ha_index += (np.random.randint(3)-1)
                new_sigma_cs_Ha_index = new_sigma_cs_Ha_index%(len(self.allowed_sigma_cs_Ha)) # don't go out of bounds
                new_sigma_cs_Ha = self.allowed_sigma_cs_Ha[new_sigma_cs_Ha_index]
                new_sigma_cs_N_index += (np.random.randint(3)-1)
                new_sigma_cs_N_index = new_sigma_cs_N_index%(len(self.allowed_sigma_cs_N)) # don't go out of bounds
                new_sigma_cs_N = self.allowed_sigma_cs_N[new_sigma_cs_N_index]
                new_sigma_cs_Ca_index += (np.random.randint(3)-1)
                new_sigma_cs_Ca_index = new_sigma_cs_Ca_index%(len(self.allowed_sigma_cs_Ca)) # don't go out of bounds
                new_sigma_cs_Ca = self.allowed_sigma_cs_Ca[new_sigma_cs_Ca_index]

            elif np.random.random() < 0.60 :        #GYH
            # take a step in array of allowed sigma_pf
                new_sigma_pf_index += (np.random.randint(3)-1)
                new_sigma_pf_index = new_sigma_pf_index%(len(self.allowed_sigma_pf)) # don't go out of bounds
                new_sigma_pf = self.allowed_sigma_pf[new_sigma_pf_index]


            elif np.random.random() < 0.78:
                # take a step in array of allowed gamma
                new_gamma_index += (np.random.randint(3)-1)
                new_gamma_index = new_gamma_index%(len(self.allowed_gamma)) # don't go out of bounds
                new_gamma  = self.allowed_gamma[new_gamma_index]



            elif np.random.random() < 0.99:
                # take a random step in state space
                new_state = np.random.randint(self.nstates)

            else:
                # pick a random pair of ambiguous groups to switch
                new_ensemble_index = np.random.randint(self.nensembles)

            # compute new "energy"
            verbose = True
#            if step%self.print(_every == 0:)
#                verbose = True
            new_E = self.neglogP(new_ensemble_index, new_state, new_sigma_noe,
                    new_sigma_J, new_sigma_cs_H, new_sigma_cs_Ha,
                    new_sigma_cs_N, new_sigma_cs_Ca, new_sigma_pf,
                    new_gamma_index,  verbose=verbose)

            # accept or reject the MC move according to Metroplis criterion
            accept = False
            if new_E < self.E:
                accept = True
            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

            # print( status)
#            if step%self.print(_every == 0:)
#                print( 'step', step, 'E', self.E, 'new_E', new_E, 'accept', accept, 'new_sigma_noe', new_sigma_noe, 'new_sigma_J', new_sigma_J, 'new_sigma_cs_H', new_sigma_cs_H, 'new_sigma_cs_Ha', new_sigma_cs_Ha, 'new_sigma_cs_N', new_sigma_cs_N, 'new_sigma_cs_Ca', new_sigma_cs_Ca, 'new_sigma_pf', new_sigma_pf, 'new_gamma', new_gamma,'new_state', new_state, 'new_ensemble_index', new_ensemble_index)
            # Store trajectory counts
            self.traj.sampled_sigma_noe[self.sigma_noe_index] += 1
            self.traj.sampled_sigma_J[self.sigma_J_index] += 1
            self.traj.sampled_sigma_cs_H[self.sigma_cs_H_index] += 1  #GYH
            self.traj.sampled_sigma_cs_Ha[self.sigma_cs_Ha_index] += 1  #GYH
            self.traj.sampled_sigma_cs_N[self.sigma_cs_N_index] += 1  #GYH
            self.traj.sampled_sigma_cs_Ca[self.sigma_cs_Ca_index] += 1  #GYH
            self.traj.sampled_sigma_pf[self.sigma_pf_index] += 1 #GYH
            self.traj.sampled_gamma[self.gamma_index] += 1
            self.traj.state_counts[self.state] += 1

            # update parameters
            if accept:
                self.E = new_E
                self.sigma_noe = new_sigma_noe
                self.sigma_noe_index = new_sigma_noe_index
                self.sigma_J = new_sigma_J
                self.sigma_J_index = new_sigma_J_index
                self.sigma_cs_H = new_sigma_cs_H                    #GYH
                self.sigma_cs_H_index = new_sigma_cs_H_index        #GYH
                self.sigma_cs_Ha = new_sigma_cs_Ha                    #GYH
                self.sigma_cs_Ha_index = new_sigma_cs_Ha_index        #GYH
                self.sigma_cs_N = new_sigma_cs_N                    #GYH
                self.sigma_cs_N_index = new_sigma_cs_N_index        #GYH
                self.sigma_cs_Ca = new_sigma_cs_Ca                    #GYH
                self.sigma_cs_Ca_index = new_sigma_cs_Ca_index        #GYH
                self.sigma_pf = new_sigma_pf                    #GYH
                self.sigma_pf_index = new_sigma_pf_index        #GYH
                self.gamma = new_gamma
                self.gamma_index = new_gamma_index
                self.state = new_state
                self.ensemble_index = new_ensemble_index
                self.accepted += 1.0
                self.total += 1.0

            # store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step), float(self.E), int(accept), int(self.state), int(self.sigma_noe_index), int(self.sigma_J_index), int(self.sigma_cs_H_index), int(self.sigma_cs_Ha_index), int(self.sigma_cs_N_index), int(self.sigma_cs_Ca_index), int(self.sigma_pf_index), int(self.gamma_index)] )    #GYH

#            if step%self.print(_every == 0:)
#                print( 'accratio =', self.accepted/self.total)
    #}}}

# }}}

# Class PosteriorSamplingTrajectory:{{{
class PosteriorSamplingTrajectory(object):
    "A class to store and perform operations on the trajectories of sampling runs."

    # __init__:{{{
    def __init__(self, ensemble, allowed_sigma_noe, allowed_sigma_J,
            allowed_sigma_cs_H, allowed_sigma_cs_Ha, allowed_sigma_cs_N,
            allowed_sigma_cs_Ca, allowed_sigma_pf, allowed_gamma):      #GYH
        "Initialize the PosteriorSamplingTrajectory."

        self.nstates = len(ensemble)
        self.ensemble = ensemble

        print( 'self.ensemble[0] = ',self.ensemble[0])
        self.ndistances = len(self.ensemble[0].distance_restraints)
        self.ndihedrals = len(self.ensemble[0].dihedral_restraints)
        self.ncs_H = len(self.ensemble[0].cs_H_restraints) #GYH
        self.ncs_Ha = len(self.ensemble[0].cs_Ha_restraints) #GYH
        self.ncs_Ca = len(self.ensemble[0].cs_Ca_restraints) #GYH
        self.ncs_N = len(self.ensemble[0].cs_N_restraints) #GYH
        self.npf = len(self.ensemble[0].pf_restraints) #GYH

        self.allowed_sigma_noe = allowed_sigma_noe
        self.sampled_sigma_noe = np.zeros(len(allowed_sigma_noe))

        self.allowed_sigma_J = allowed_sigma_J
        self.sampled_sigma_J = np.zeros(len(allowed_sigma_J))

        self.allowed_sigma_cs_H = allowed_sigma_cs_H                        #GYH
        self.sampled_sigma_cs_H = np.zeros(len(allowed_sigma_cs_H))         #GYH

        self.allowed_sigma_cs_Ha = allowed_sigma_cs_Ha                        #GYH
        self.sampled_sigma_cs_Ha = np.zeros(len(allowed_sigma_cs_Ha))         #GYH

        self.allowed_sigma_cs_N = allowed_sigma_cs_N                        #GYH
        self.sampled_sigma_cs_N = np.zeros(len(allowed_sigma_cs_N))         #GYH

        self.allowed_sigma_cs_Ca = allowed_sigma_cs_Ca                        #GYH
        self.sampled_sigma_cs_Ca = np.zeros(len(allowed_sigma_cs_Ca))         #GYH

        self.allowed_sigma_pf = allowed_sigma_pf                        #GYH
        self.sampled_sigma_pf = np.zeros(len(allowed_sigma_pf))         #GYH

        self.allowed_gamma = allowed_gamma
        self.sampled_gamma = np.zeros(len(allowed_gamma))


        self.state_counts = np.ones(self.nstates)  # add a pseudocount to avoid log(0) errors

        self.f_sim = np.array([e.free_energy for e in ensemble])
        self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()

        # stores samples [step, self.E, accept, state, sigma_noe, sigma_J, sigma_cs, gamma]
        self.trajectory_headers = ['step', 'E', 'accept', 'state', 'sigma_noe_index', 'sigma_J_index', 'sigma_cs_H_index', 'sigma_cs_Ha_index', 'sigma_cs_N_index', 'sigma_cs_Ca_index', 'sigma_pf_index', 'gamma_index']       #GYH
        self.trajectory = []

        # a dictionary to store results for YAML file
        self.results = {}

    #}}}

    # Process the Trajectory:{{{
    def process(self):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables.

        NOTE: Where possible, we convert to lists, because the YAML output
        is more readable"""

        # Store the trajectory in rsults
        self.results['trajectory_headers'] = self.trajectory_headers
        self.results['trajectory'] = self.trajectory

        # Store the nuisance parameter distributions
        self.results['allowed_sigma_noe'] = self.allowed_sigma_noe.tolist()
        self.results['allowed_sigma_J'] = self.allowed_sigma_J.tolist()
        self.results['allowed_sigma_cs_H'] = self.allowed_sigma_cs_H.tolist()   #GYH
        self.results['allowed_sigma_cs_Ha'] = self.allowed_sigma_cs_Ha.tolist()   #GYH
        self.results['allowed_sigma_cs_N'] = self.allowed_sigma_cs_N.tolist()   #GYH
        self.results['allowed_sigma_cs_Ca'] = self.allowed_sigma_cs_Ca.tolist()   #GYH
        self.results['allowed_sigma_pf'] = self.allowed_sigma_pf.tolist() #GYH
        self.results['allowed_gamma'] = self.allowed_gamma.tolist()
        self.results['sampled_sigma_noe'] = self.sampled_sigma_noe.tolist()
        self.results['sampled_sigma_J'] = self.sampled_sigma_J.tolist()
        self.results['sampled_sigma_cs_H'] = self.sampled_sigma_cs_H.tolist()   #GYH
        self.results['sampled_sigma_cs_Ha'] = self.sampled_sigma_cs_Ha.tolist()   #GYH
        self.results['sampled_sigma_cs_N'] = self.sampled_sigma_cs_N.tolist()   #GYH
        self.results['sampled_sigma_cs_Ca'] = self.sampled_sigma_cs_Ca.tolist()   #GYH
        self.results['sampled_sigma_pf'] = self.sampled_sigma_pf.tolist()   #GYH
        self.results['sampled_gamma'] = self.sampled_gamma.tolist()

        # Calculate the modes of the nuisance parameter marginal distributions
        self.results['sigma_noe_mode'] = float(self.allowed_sigma_noe[ np.argmax(self.sampled_sigma_noe) ])
        self.results['sigma_J_mode']   = float(self.allowed_sigma_J[ np.argmax(self.sampled_sigma_J) ])
        self.results['sigma_cs_H_mode']   = float(self.allowed_sigma_cs_H[ np.argmax(self.sampled_sigma_cs_H) ])      #GYH
        self.results['sigma_cs_Ha_mode']   = float(self.allowed_sigma_cs_Ha[ np.argmax(self.sampled_sigma_cs_Ha) ])      #GYH
        self.results['sigma_cs_N_mode']   = float(self.allowed_sigma_cs_N[ np.argmax(self.sampled_sigma_cs_N) ])      #GYH
        self.results['sigma_cs_Ca_mode']   = float(self.allowed_sigma_cs_Ca[ np.argmax(self.sampled_sigma_cs_Ca) ])      #GYH
        self.results['sigma_pf_mode']       = float(self.allowed_sigma_pf[ np.argmax(self.sampled_sigma_pf) ])    #GYH
        self.results['gamma_mode']     = float(self.allowed_gamma[ np.argmax(self.sampled_gamma) ])

        # copy over the purely computational free energies f_i
        self.results['comp_f'] = self.f_sim.tolist()

        # Estimate the populations of each state
        self.results['state_pops'] = (self.state_counts/self.state_counts.sum()).tolist()

        # Estimate uncertainty in the populations by bootstrap
        self.nbootstraps = 1000
        self.bootstrapped_state_pops = np.random.multinomial(self.state_counts.sum(), self.results['state_pops'], size=self.nbootstraps)
        self.results['state_pops_std'] = self.bootstrapped_state_pops.std(axis=0).tolist()

        # Estimate the free energies of each state
        self.results['state_f'] = (-np.log(self.results['state_pops'])).tolist()
        state_f = -np.log(self.results['state_pops'])
        ref_f = state_f.min()
        state_f -=  ref_f
        self.results['state_f'] = state_f.tolist()
        self.bootstrapped_state_f = -np.log(self.bootstrapped_state_pops+1e-10) - ref_f  # add pseudocount to avoid log(0)s in the bootstrap
        self.results['state_f_std'] = self.bootstrapped_state_f.std(axis=0).tolist()

        # Estimate the ensemble-<r**-6>averaged distances
        mean_distances = np.zeros(self.ndistances)
        Z = np.zeros(self.ndistances)
        for i in range(self.nstates):
            for j in range(self.ndistances):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].distance_restraints[j].weight
                r = self.ensemble[i].distance_restraints[j].model_distance
                mean_distances[j] += pop*weight*(r**(-6.0))
                Z[j] += pop*weight
        mean_distances = (mean_distances/Z)**(-1.0/6.0)
        self.results['mean_distances'] = mean_distances.tolist()

        # compute the experimental distances, using the most likely gamma'
        exp_distances = np.array([self.results['gamma_mode']*self.ensemble[0].distance_restraints[j].exp_distance \
                                      for j in range(self.ndistances)])
        self.results['exp_distances'] = exp_distances.tolist()

        self.results['distance_pairs'] = []
        for j in range(self.ndistances):
            pair = [int(self.ensemble[0].distance_restraints[j].i), int(self.ensemble[0].distance_restraints[j].j)]
            self.results['distance_pairs'].append(pair)
        abs_diffs = np.abs( exp_distances - mean_distances )
        self.results['disagreement_distances_mean'] = float(abs_diffs.mean())
        self.results['disagreement_distances_std'] = float(abs_diffs.std())

        # Estimate the ensemble-averaged J-coupling values
        mean_Jcoupling = np.zeros(self.ndihedrals)
        Z = np.zeros(self.ndihedrals)
        for i in range(self.nstates):
            for j in range(self.ndihedrals):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].dihedral_restraints[j].weight
                r = self.ensemble[i].dihedral_restraints[j].model_Jcoupling
                mean_Jcoupling[j] += pop*weight*r
                Z[j] += pop*weight
        mean_Jcoupling = (mean_Jcoupling/Z)     #GYH
        self.results['mean_Jcoupling'] = mean_Jcoupling.tolist()

        # Compute the experiment Jcouplings
        exp_Jcoupling = np.array([self.ensemble[0].dihedral_restraints[j].exp_Jcoupling for j in range(self.ndihedrals)])
        self.results['exp_Jcoupling'] = exp_Jcoupling.tolist()
        abs_Jdiffs = np.abs( exp_Jcoupling - mean_Jcoupling )
        self.results['disagreement_Jcoupling_mean'] = float(abs_Jdiffs.mean())
        self.results['disagreement_Jcoupling_std'] = float(abs_Jdiffs.std())

        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_cs_H = np.zeros(self.ncs_H)
        Z = np.zeros(self.ncs_H)
        for i in range(self.nstates):
            for j in range(self.ncs_H):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].cs_H_restraints[j].weight
                r = self.ensemble[i].cs_H_restraints[j].model_cs_H
                mean_cs_H[j] += pop*weight*r
                Z[j] += pop*weight
        mean_cs_H = (mean_cs_H/Z)
        self.results['mean_cs_H'] = mean_cs_H.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_cs_H = np.array([self.ensemble[0].cs_H_restraints[j].exp_cs_H for j in range(self.ncs_H)])
        self.results['exp_cs_H'] = exp_cs_H.tolist()
        abs_cs_H_diffs = np.abs( exp_cs_H - mean_cs_H )
        self.results['disagreement_cs_H_mean'] = float(abs_cs_H_diffs.mean())
        self.results['disagreement_cs_H_std'] = float(abs_cs_H_diffs.std())

        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_cs_Ha = np.zeros(self.ncs_Ha)
        Z = np.zeros(self.ncs_Ha)
        for i in range(self.nstates):
            for j in range(self.ncs_Ha):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].cs_Ha_restraints[j].weight
                r = self.ensemble[i].cs_Ha_restraints[j].model_cs_Ha
                mean_cs_Ha[j] += pop*weight*r
                Z[j] += pop*weight
        mean_cs_Ha = (mean_cs_Ha/Z)
        self.results['mean_cs_Ha'] = mean_cs_Ha.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_cs_Ha = np.array([self.ensemble[0].cs_Ha_restraints[j].exp_cs_Ha for j in range(self.ncs_Ha)])
        self.results['exp_cs_Ha'] = exp_cs_Ha.tolist()
        abs_cs_Ha_diffs = np.abs( exp_cs_Ha - mean_cs_Ha )
        self.results['disagreement_cs_Ha_mean'] = float(abs_cs_Ha_diffs.mean())
        self.results['disagreement_cs_Ha_std'] = float(abs_cs_Ha_diffs.std())


        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_cs_N = np.zeros(self.ncs_N)
        Z = np.zeros(self.ncs_N)
        for i in range(self.nstates):
            for j in range(self.ncs_N):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].cs_N_restraints[j].weight
                r = self.ensemble[i].cs_N_restraints[j].model_cs_N
                mean_cs_N[j] += pop*weight*r
                Z[j] += pop*weight
        mean_cs_N = (mean_cs_N/Z)
        self.results['mean_cs_N'] = mean_cs_N.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_cs_N = np.array([self.ensemble[0].cs_N_restraints[j].exp_cs_N for j in range(self.ncs_N)])
        self.results['exp_cs_N'] = exp_cs_N.tolist()
        abs_cs_N_diffs = np.abs( exp_cs_N - mean_cs_N )
        self.results['disagreement_cs_N_mean'] = float(abs_cs_N_diffs.mean())
        self.results['disagreement_cs_N_std'] = float(abs_cs_N_diffs.std())


        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_cs_Ca = np.zeros(self.ncs_Ca)
        Z = np.zeros(self.ncs_Ca)
        for i in range(self.nstates):
            for j in range(self.ncs_Ca):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].cs_Ca_restraints[j].weight
                r = self.ensemble[i].cs_Ca_restraints[j].model_cs_Ca
                mean_cs_Ca[j] += pop*weight*r
                Z[j] += pop*weight
        mean_cs_Ca = (mean_cs_Ca/Z)
        self.results['mean_cs_Ca'] = mean_cs_Ca.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_cs_Ca = np.array([self.ensemble[0].cs_Ca_restraints[j].exp_cs_Ca for j in range(self.ncs_Ca)])
        self.results['exp_cs_Ca'] = exp_cs_Ca.tolist()
        abs_cs_Ca_diffs = np.abs( exp_cs_Ca - mean_cs_Ca )
        self.results['disagreement_cs_Ca_mean'] = float(abs_cs_Ca_diffs.mean())
        self.results['disagreement_cs_Ca_std'] = float(abs_cs_Ca_diffs.std())



        # Estimate the ensemble-averaged protection factor values    #GYH
        mean_pf = np.zeros(self.npf)
        Z = np.zeros(self.npf)
        for i in range(self.nstates):
            for j in range(self.npf):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].pf_restraints[j].weight
                r = self.ensemble[i].pf_restraints[j].model_pf
                mean_pf[j] += pop*weight*r
                Z[j] += pop*weight
        mean_pf = (mean_pf/Z)    #GYH
        self.results['mean_pf'] = mean_pf.tolist()

        # Compute the experiment protection factor                 #GYH

        exp_pf = np.array([self.ensemble[0].pf_restraints[j].exp_pf for j in range(self.npf)])
#        exp_pf = np.array([self.ensemble[0].pf_restraints[j].exp_pf for j in range(self.npf)])
        self.results['exp_pf'] = exp_pf.tolist()
        abs_pfdiffs = np.abs( exp_pf - mean_pf )
        self.results['disagreement_pf_mean'] = float(abs_pfdiffs.mean())
        self.results['disagreement_pf_std'] = float(abs_pfdiffs.std())
    #}}}

    # log Spaced Array:{{{
    def logspaced_array(self, xmin, xmax, nsteps):
        ymin, ymax = np.log(xmin), np.log(xmax)
        dy = (ymax-ymin)/nsteps
        return np.exp(np.arange(ymin, ymax, dy))
    #}}}

    ## Write Results:{{{
    #def write_results(self, outfilename='traj.yaml'):
    #    """Dumps results to a YAML format file. """

    #    # Read in the YAML data as a dictionary
    #    fout = file(outfilename, 'w')
    #    yaml.dump(self.results, fout, default_flow_style=False)
    ##}}}

    #import numpy as np
    #import yaml, io
    #import h5py
    #import pickle
    #import xml


#    # Numpy Z Compression{{{
#    #NOTE: This will work well with Cython if we go that route.
#    # Standardized: Yes ; Binary: Yes; Human Readable: No;
#
    def write_results(self, outfilename):
        """Writes a compact file of several arrays into binary format."""

        np.savez_compressed(outfilename, self.results)

#    def read_results(filename):
#        """Reads a npz file"""
#
#        loaded = np.load(filename)
#        print( loaded.items())
#    # }}}
#
#    # YAML{{{
#    #NOTE:
#    # Standardized: Yes; Binary: No; Human Readable: Yes;
#
#    def write_results(self, outfilename):
#        """Dumps results to a YAML format file. """
#
#        fout = file(outfilename, 'w')
#        yaml.dump(self.results, fout, default_flow_style=False)
#
#    def read_results(filename):
#        '''Reads a yaml file.'''
#
#        with io.open(filename,'r') as file:
#            loaded_data = yaml.load(file)
#            print(('%s'%loaded_data).replace(" '","\n\n '"))
## }}}
#
#    # H5{{{
#    #NOTE: Cython wrapping of the HDF5 C API
#    # Standardized: Yes; Binary: Yes; Human Readable: No;
#
#    def write_results(self, outfilename):
#        """ """
#
#        hf = h5py.File(outfilename, 'a')
#        for k,v in self.results.items():
#            hf.create_dataset(k, data=v)
#        hf.close()
#
#
#    def write_results(self, outfilename):
#        """ """
#
#        hf = h5py.File(outfilename, 'w')
#        grp = hf.create_group(None)
#        for k,v in self.results.items():
#            grp.create_dataset(k,data=v)
#        hf.create_dataset('dataset', data=self.results)
#        hf.close()





    def read_results(filename):
        f = h5py.File(filename,'r')
        print( f.items())
        #a_group_key = list(f.keys())[0]
        # Get the data
        #data = list(f[a_group_key])
# }}}
#
#    # Python Pickle{{{
#    #NOTE:
#    # Standardized: Yes; Binary: Yes; Human Readable: No;
#
#    def write_results(self, outfilename):
#        """Writes results as a pickle file."""
#
#        pkl = open(outfilename, 'wb')
#        pickle.dump(self.results, pkl)
#        pkl.close()
#
#    def read_results(filename):
#        """ """
#
#        pkl = open(filename, 'r')
#        loaded = pickle.load(pkl)
#        print( loaded)
#
## }}}
#






#}}}





