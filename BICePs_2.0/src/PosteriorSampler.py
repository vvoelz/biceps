#!/usr/bin/env python

# Import Modules:{{{
import os, sys, glob, copy
import numpy as np
from scipy  import loadtxt, savetxt
from matplotlib import pylab as plt
import yaml
from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from RestraintFile_cs import *    # Class - creates Chemical shift restraint file
from RestraintFile_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from RestraintFile_J import *     # Class - creates J-coupling const. restraint file
from RestraintFile_pf import *	  # Class - creates Protection factor restraint file   #GYH
from restraint_cs_H import *	# test (until compute sse) done for work   Yunhui Ge -- 04/2018
from restraint_J import *
from restraint_cs_Ha import *
from restraint_cs_N import *
from restraint_cs_CA import *
from restraint_noe import *



# }}}

# Class PosteriorSampler: {{{
class PosteriorSampler(object):
    """A class to perform posterior sampling of conformational populations"""

    # __init__:{{{
    def __init__(self, ensemble, data = None,
            dlogsigma_noe=np.log(1.01), sigma_noe_min=0.05, sigma_noe_max=120.0,
            dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
            dlogsigma_cs_H=np.log(1.02),sigma_cs_H_min=0.05, sigma_cs_H_max=20.0,
	    dlogsigma_cs_Ha=np.log(1.02),sigma_cs_Ha_min=0.05, sigma_cs_Ha_max=20.0,
	    dlogsigma_cs_N=np.log(1.02),sigma_cs_N_min=0.05, sigma_cs_N_max=20.0,
	    dlogsigma_cs_Ca=np.log(1.02),sigma_cs_Ca_min=0.05, sigma_cs_Ca_max=20.0,
	    dlogsigma_PF=np.log(1.02),sigma_PF_min=0.05, sigma_PF_max=20.0,	#GYH
	    dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=10.0,
            dalpha=0.1, alpha_min=-5.0, alpha_max=5.0,
            distribution='exponential',
            ambiguous_groups=False,
            sample_ambiguous_distances=True,
	    use_reference_prior_noe=False,
            use_reference_prior_H=False,
            use_reference_prior_Ha=False,
            use_reference_prior_N=False,
            use_reference_prior_Ca=False,
            use_reference_prior_PF=False,
            use_gaussian_reference_prior_noe=False,
            use_gaussian_reference_prior_H=False,
            use_gaussian_reference_prior_Ha=False,
            use_gaussian_reference_prior_N=False,
            use_gaussian_reference_prior_Ca=False,
            use_gaussian_reference_prior_PF=False):	#GYH
        "Initialize the PosteriorSampler class."

        # Notes on ## Ensemble ##:
        '''
        The ensemble is a list that is appended by Structure objects:
        recall: Structure(self, gens [pdbs], free energy,  expdata_filename_noe,
                   use_log_normal_distances,
            dloggamma=np.log(1.01), gamma_min=0.2,
            gamma_max=5.0, dalpha=0.1, alpha_min=0.0, alpha_max=0.1)


        s = Structure('Gens/micro/%d.pdb'%i, args.lam*energies[i],
            expdata_filename_noe=expdata_filename_noe,
            use_log_normal_distances=False,
            dloggamma=np.log(1.01), gamma_min=0.2,
            gamma_max=5.0, dalpha=0.1, alpha_min=0.0, alpha_max=0.1)


        '''

        # the ensemble is a list of Structure() objects
        self.ensembles = [ ensemble ]   # why a list inside of a list?
        self.nstates = len(ensemble)    # number of structures (elements)
        self.nensembles = len(self.ensembles)  # if [ ensemble ], then this should
                                               # always be equal to 1
        self.ensemble_index = 0

        # need to keep track of ambiguous distances and multiple ensembles to sample over
        self.ambiguous_groups = []  # list of pairs of group indices, e.g.:   [ [[1,2,3],[4,5,6]],   [[7],[8]], ...]
        #self.ambiguous_groups = ensemble[0].ambiguous_groups
        self.sample_ambiguous_distances = sample_ambiguous_distances
        if sample_ambiguous_distances:
            self.build_alternative_ensembles()


        self.distribution = distribution #RMR

        # pick initial values for sigma_noe (std of experimental uncertainty in NOE distances)
        self.dlogsigma_noe = dlogsigma_noe  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_noe_min = sigma_noe_min
        self.sigma_noe_max = sigma_noe_max
        self.allowed_sigma_noe = np.exp(np.arange(np.log(self.sigma_noe_min), np.log(self.sigma_noe_max), self.dlogsigma_noe))
#        print 'self.allowed_sigma_noe', self.allowed_sigma_noe
#        print 'len(self.allowed_sigma_noe) =', len(self.allowed_sigma_noe)
        self.sigma_noe_index = len(self.allowed_sigma_noe)/2    # pick an intermediate value to start with
        self.sigma_noe = self.allowed_sigma_noe[self.sigma_noe_index]

        # pick initial values for sigma_J (std of experimental uncertainty in J-coupling constant)
        self.dlogsigma_J = dlogsigma_J  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_J_min = sigma_J_min
        self.sigma_J_max = sigma_J_max
        self.allowed_sigma_J = np.exp(np.arange(np.log(self.sigma_J_min), np.log(self.sigma_J_max), self.dlogsigma_J))
#        print 'self.allowed_sigma_J', self.allowed_sigma_J
#        print 'len(self.allowed_sigma_J) =', len(self.allowed_sigma_J)

        self.sigma_J_index = len(self.allowed_sigma_J)/2   # pick an intermediate value to start with
        self.sigma_J = self.allowed_sigma_J[self.sigma_J_index]


	# pick initial values for sigma_cs_H (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_H = dlogsigma_cs_H  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_H_min = sigma_cs_H_min
        self.sigma_cs_H_max = sigma_cs_H_max
        self.allowed_sigma_cs_H = np.exp(np.arange(np.log(self.sigma_cs_H_min), np.log(self.sigma_cs_H_max), self.dlogsigma_cs_H))
#        print 'self.allowed_sigma_cs_H', self.allowed_sigma_cs_H
#        print 'len(self.allowed_sigma_cs_H) =', len(self.allowed_sigma_cs_H)
        self.sigma_cs_H_index = len(self.allowed_sigma_cs_H)/2   # pick an intermediate value to start with
        self.sigma_cs_H = self.allowed_sigma_cs_H[self.sigma_cs_H_index]

        # pick initial values for sigma_cs_Ha (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ha = dlogsigma_cs_Ha  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ha_min = sigma_cs_Ha_min
        self.sigma_cs_Ha_max = sigma_cs_Ha_max
        self.allowed_sigma_cs_Ha = np.exp(np.arange(np.log(self.sigma_cs_Ha_min), np.log(self.sigma_cs_Ha_max), self.dlogsigma_cs_Ha))
#        print 'self.allowed_sigma_cs_Ha', self.allowed_sigma_cs_Ha
#        print 'len(self.allowed_sigma_cs_Ha) =', len(self.allowed_sigma_cs_Ha)
        self.sigma_cs_Ha_index = len(self.allowed_sigma_cs_Ha)/2   # pick an intermediate value to start with
        self.sigma_cs_Ha = self.allowed_sigma_cs_Ha[self.sigma_cs_Ha_index]


        # pick initial values for sigma_cs_N (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_N = dlogsigma_cs_N  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_N_min = sigma_cs_N_min
        self.sigma_cs_N_max = sigma_cs_N_max
        self.allowed_sigma_cs_N = np.exp(np.arange(np.log(self.sigma_cs_N_min), np.log(self.sigma_cs_N_max), self.dlogsigma_cs_N))
#        print 'self.allowed_sigma_cs_N', self.allowed_sigma_cs_N
#        print 'len(self.allowed_sigma_cs_N) =', len(self.allowed_sigma_cs_N)
        self.sigma_cs_N_index = len(self.allowed_sigma_cs_N)/2   # pick an intermediate value to start with
        self.sigma_cs_N = self.allowed_sigma_cs_N[self.sigma_cs_N_index]

        # pick initial values for sigma_cs_Ca (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_cs_Ca = dlogsigma_cs_Ca  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_cs_Ca_min = sigma_cs_Ca_min
        self.sigma_cs_Ca_max = sigma_cs_Ca_max
        self.allowed_sigma_cs_Ca = np.exp(np.arange(np.log(self.sigma_cs_Ca_min), np.log(self.sigma_cs_Ca_max), self.dlogsigma_cs_Ca))
#        print 'self.allowed_sigma_cs_Ca', self.allowed_sigma_cs_Ca
#        print 'len(self.allowed_sigma_cs_Ca) =', len(self.allowed_sigma_cs_Ca)
        self.sigma_cs_Ca_index = len(self.allowed_sigma_cs_Ca)/2   # pick an intermediate value to start with
        self.sigma_cs_Ca = self.allowed_sigma_cs_Ca[self.sigma_cs_Ca_index]



        # pick initial values for sigma_PF (std of experimental uncertainty in chemical shift)   #GYH
        self.dlogsigma_PF = dlogsigma_PF  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_PF_min = sigma_PF_min
        self.sigma_PF_max = sigma_PF_max
        self.allowed_sigma_PF = np.exp(np.arange(np.log(self.sigma_PF_min), np.log(self.sigma_PF_max), self.dlogsigma_PF))
#        print 'self.allowed_sigma_PF', self.allowed_sigma_PF
#        print 'len(self.allowed_sigma_PF) =', len(self.allowed_sigma_PF)
        self.sigma_PF_index = len(self.allowed_sigma_PF)/2   # pick an intermediate value to start with
        self.sigma_PF = self.allowed_sigma_PF[self.sigma_PF_index]



        # pick initial values for gamma^(-1/6) (NOE scaling parameter)
        self.dloggamma = dloggamma  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
#        print 'self.allowed_gamma', self.allowed_gamma
#        print 'len(self.allowed_gamma) =', len(self.allowed_gamma)
        self.gamma_index = len(self.allowed_gamma)/2    # pick an intermediate value to start with
        self.gamma = self.allowed_gamma[self.gamma_index]

        # pick initial values for alpha (PF 'scaling' parameter)	#GYH
        self.dalpha = dalpha  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.allowed_alpha = np.arange(self.alpha_min, self.alpha_max, self.dalpha)
#        print 'self.allowed_alpha', self.allowed_alpha
#        print 'len(self.allowed_alpha) =', len(self.allowed_alpha)
        self.alpha_index = len(self.allowed_alpha)/2    # pick an intermediate value to start with
        self.alpha = self.allowed_alpha[self.alpha_index]

        # the initial state of the structural ensemble we're sampling from
        self.state = 0    # index in the ensemble
        self.E = 1.0e99   # initial energy
        self.accepted = 0
        self.total = 0

        # keep track of what we sampled in a trajectory
        self.traj = PosteriorSamplingTrajectory(self.ensembles[0],
                self.allowed_sigma_noe,
                self.allowed_sigma_J,
                self.allowed_sigma_cs_H,
                self.allowed_sigma_cs_Ha,
                self.allowed_sigma_cs_N,
                self.allowed_sigma_cs_Ca,
                self.allowed_sigma_PF,
                self.allowed_gamma,
                self.allowed_alpha)
        self.write_traj = 1000  # step frequencies to write trajectory info

        # frequency of printing to the screen
        self.print_every = 1000 # debug

        # frequency of storing trajectory samples
        self.traj_every = 100

        # compile reference prior of distances from the uniform distribution of distances
        self.use_reference_prior_noe = use_reference_prior_noe
        if self.use_reference_prior_noe:
            self.build_reference_prior_noe()
	self.use_reference_prior_H = use_reference_prior_H
        if self.use_reference_prior_H:
            self.build_reference_prior_H()
	self.use_reference_prior_Ha = use_reference_prior_Ha
        if self.use_reference_prior_Ha:
            self.build_reference_prior_Ha()
	self.use_reference_prior_N = use_reference_prior_N
        if self.use_reference_prior_N:
            self.build_reference_prior_N()
	self.use_reference_prior_Ca = use_reference_prior_Ca
        if self.use_reference_prior_Ca:
            self.build_reference_prior_Ca()
	self.use_reference_prior_PF = use_reference_prior_PF
        if self.use_reference_prior_PF:
            self.build_reference_prior_PF()

        self.use_gaussian_reference_prior_noe = use_gaussian_reference_prior_noe
        if self.use_gaussian_reference_prior_noe:
            self.build_gaussian_reference_prior_noe()
        self.use_gaussian_reference_prior_H = use_gaussian_reference_prior_H
        if self.use_gaussian_reference_prior_H:
            self.build_gaussian_reference_prior_H()
        self.use_gaussian_reference_prior_Ha = use_gaussian_reference_prior_Ha
        if self.use_gaussian_reference_prior_Ha:
            self.build_gaussian_reference_prior_Ha()
        self.use_gaussian_reference_prior_N = use_gaussian_reference_prior_N
        if self.use_gaussian_reference_prior_N:
            self.build_gaussian_reference_prior_N()
        self.use_gaussian_reference_prior_Ca = use_gaussian_reference_prior_Ca
        if self.use_gaussian_reference_prior_Ca:
            self.build_gaussian_reference_prior_Ca()
        self.use_gaussian_reference_prior_PF = use_gaussian_reference_prior_PF
        if self.use_gaussian_reference_prior_PF:
            self.build_gaussian_reference_prior_PF()

	   # make a flag if using both exp and gaussian ref potentials for one restraint
        if self.use_reference_prior_noe and self.use_gaussian_reference_prior_noe:
                sys.exit("error: Cannot use different reference potentials for one exp restraint")
        if self.use_reference_prior_H and self.use_gaussian_reference_prior_H:
                sys.exit("error: Cannot use different reference potentials for one exp restraint")
        if self.use_reference_prior_Ha and self.use_gaussian_reference_prior_Ha:
                sys.exit("error: Cannot use different reference potentials for one exp restraint")
        if self.use_reference_prior_N and self.use_gaussian_reference_prior_N:
                sys.exit("error: Cannot use different reference potentials for one exp restraint")
        if self.use_reference_prior_Ca and self.use_gaussian_reference_prior_Ca:
                sys.exit("error: Cannot use different reference potentials for one exp restraint")
  	if self.use_reference_prior_PF and self.use_gaussian_reference_prior_PF:
	        sys.exit("error: Cannot use different reference potentials for one exp restraint")


        # RMR:
	r_cs_H = restraint_cs_H()
        r_cs_Ha = restraint_cs_Ha()
        r_cs_N = restraint_cs_N()
        r_cs_CA = restraint_cs_CA()
	r_J = restraint_J()
	r_noe = restraint_noe()

#        # When using self.distribution:{{{
#	if data != None:
#	    for i in data:
#	        if i.endswith('.noe'):
#                    r_noe.load_data_noe(i)
#                    if self.distribution == 'exponential':
#                        print 'Using Exponential Distribution'
#                        use_reference_prior_H=True
#                    elif self.distribution == 'gaussian':
#                        print 'Using Gaussian Distribution'
#                        use_gaussian_reference_prior_H=True
#                    else:
#                        use_reference_prior_H=False
#                        use_gaussian_reference_prior_H=False
#	    	elif i.endswith('.J'):
#                    r_J.load_data_J(i)
#                    if self.distribution == 'exponential':
#                        print 'Using Exponential Distribution'
#                        use_reference_prior_H=True
#                    elif self.distribution == 'gaussian':
#                        print 'Using Gaussian Distribution'
#                        use_gaussian_reference_prior_H=True
#                    else:
#                        use_reference_prior_H=False
#                        use_gaussian_reference_prior_H=False
#                elif i.endswith('.cs_H'):
#                    r_cs_H.load_data_cs_H(i)
#                    if self.distribution == 'exponential':
#                        print 'Using Exponential Distribution'
#                        use_reference_prior_H=True
#                    elif self.distribution == 'gaussian':
#                        print 'Using Gaussian Distribution'
#                        use_gaussian_reference_prior_H=True
#                    else:
#                        use_reference_prior_H=False
#                        use_gaussian_reference_prior_H=False
#                elif i.endswith('.cs_Ha'):
#                    r_cs_Ha.load_data_cs_Ha(i)
#                    if self.distribution == 'exponential':
#                        print 'Using Exponential Distribution'
#                        use_reference_prior_H=True
#                    elif self.distribution == 'gaussian':
#                        print 'Using Gaussian Distribution'
#                        use_gaussian_reference_prior_H=True
#                    else:
#                        use_reference_prior_H=False
#                        use_gaussian_reference_prior_H=False
#                elif i.endswith('.cs_N'):
#                    r_cs_N.load_data_cs_N(i)
#                    if self.distribution == 'exponential':
#                        print 'Using Exponential Distribution'
#                        use_reference_prior_H=True
#                    elif self.distribution == 'gaussian':
#                        print 'Using Gaussian Distribution'
#                        use_gaussian_reference_prior_H=True
#                    else:
#                        use_reference_prior_H=False
#                        use_gaussian_reference_prior_H=False
#                elif i.endswith('.cs_CA'):
#                    r_cs_CA.load_data_cs_CA(i)
#                    if self.distribution == 'exponential':
#                        print 'Using Exponential Distribution'
#                        use_reference_prior_H=True
#                    elif self.distribution == 'gaussian':
#                        print 'Using Gaussian Distribution'
#                        use_gaussian_reference_prior_H=True
#                else:
#                    raise ValueError("Incompatible File extension. Use:{'.noe','.J','.cs_H','.cs_Ha'}")
#	else:
#	    raise ValueError("Something is wrong in your input file (necessary input file missing)")
## }}}
#
        # If there exists different exstensions in the data input:{{{
        # RMR:

        dist_prompt = 'What type of distribution would you like for your %s data?\n\
                        Type the number, then press enter.\n\
                        1. Exponential\n\
                        2. Gaussian\n\
                        3. Uniform\n\
                        '
        wd = os.path.dirname(os.path.realpath(data[0]))
        #print wd
        # specifying the location of config file
        cs_H_Log  = wd+'/.cs_H.log'
        cs_Ha_Log = wd+'/.cs_Ha.log'
        cs_N_Log  = wd+'/.cs_N.log'
        cs_CA_Log = wd+'/.cs_CA.log'
        J_Log     = wd+'/.J.log'
        noe_Log   = wd+'/.noe.log'
        if data != None:
	    for i in data:
	        if i.endswith('.noe'):
                    if os.path.isfile(noe_Log): # storing a log file to contain configuration of distribution settings
                        config = open(noe_Log, 'r')
                        distribution = config.readline()
                        #os.system('echo %s'%dist)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_noe=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_noe=True
                        else:
                            use_reference_prior_noe=False
                            use_gaussian_reference_prior_noe=False
                    else:
                        distribution = input(dist_prompt%'noe')
                        with open('%s'%noe_Log, 'w') as f:
                            f.write('%d'%distribution)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_noe=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_noe=True
                        else:
                            use_reference_prior_noe=False
                            use_gaussian_reference_prior_noe=False


	    	elif i.endswith('.J'):
                    if os.path.isfile(J_Log): # storing a log file to contain configuration of distribution settings
                        config = open(J_Log, 'r')
                        dist = config.readline()
                        #os.system('echo %s'%dist)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_J=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_J=True
                        else:
                            use_reference_prior_J=False
                            use_gaussian_reference_prior_J=False
                    else:
                        distribution = input(dist_prompt%'J')
                        with open('%s'%J_Log, 'w') as f:
                            f.write('%d'%distribution)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_J=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_J=True
                        else:
                            use_reference_prior_J=False
                            use_gaussian_reference_prior_J=False


                elif i.endswith('.cs_H'):
                    if os.path.isfile(cs_H_Log): # storing a log file to contain configuration of distribution settings
                        config = open(cs_H_Log, 'r')
                        dist = config.readline()
                        #os.system('echo %s'%dist)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_H=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_H=True
                        else:
                            use_reference_prior_H=False
                            use_gaussian_reference_prior_H=False
                    else:
                        distribution = input(dist_prompt%'H')
                        with open('%s'%cs_H_Log, 'w') as f:
                            f.write('%d'%distribution)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_H=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_H=True
                        else:
                            use_reference_prior_H=False
                            use_gaussian_reference_prior_H=False


                elif i.endswith('.cs_Ha'):
                    if os.path.isfile(cs_Ha_Log): # storing a log file to contain configuration of distribution settings
                        config = open(cs_Ha_Log, 'r')
                        dist = config.readline()
                        #os.system('echo %s'%dist)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_Ha=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_Ha=True
                        else:
                            use_reference_prior_Ha=False
                            use_gaussian_reference_prior_Ha=False
                    else:
                        distribution = input(dist_prompt%'Ha')
                        with open('%s'%cs_Ha_Log, 'w') as f:
                            f.write('%d'%distribution)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_Ha=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_Ha=True
                        else:
                            use_reference_prior_Ha=False
                            use_gaussian_reference_prior_Ha=False

                elif i.endswith('.cs_N'):
                    if os.path.isfile(cs_N_Log): # storing a log file to contain configuration of distribution settings
                        config = open(cs_N_Log, 'r')
                        dist = config.readline()
                        #os.system('echo %s'%dist)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_N=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_N=True
                        else:
                            use_reference_prior_N=False
                            use_gaussian_reference_prior_N=False
                    else:
                        distribution = input(dist_prompt%'N')
                        with open('%s'%cs_N_Log, 'w') as f:
                            f.write('%d'%distribution)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_N=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_N=True
                        else:
                            use_reference_prior_N=False
                            use_gaussian_reference_prior_N=False

                elif i.endswith('.cs_CA'):
                    if os.path.isfile(cs_CA_Log): # storing a log file to contain configuration of distribution settings
                        config = open(cs_CA_Log, 'r')
                        dist = config.readline()
                        #os.system('echo %s'%dist)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_Ca=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_Ca=True
                        else:
                            use_reference_prior_Ca=False
                            use_gaussian_reference_prior_Ca=False
                    else:
                        distribution = input(dist_prompt%'Ca')
                        with open('%s'%cs_Ca_Log, 'w') as f:
                            f.write('%d'%distribution)
                        if distribution == 1:
                            print 'Using Exponential Distribution'
                            use_reference_prior_Ca=True
                        elif distribution == 2:
                            print 'Using Gaussian Distribution'
                            use_gaussian_reference_prior_Ca=True
                        else:
                            use_reference_prior_Ca=False
                            use_gaussian_reference_prior_Ca=False

                else:
                    raise ValueError("Incompatible File extension. Use:{'.noe','.J','.cs_H','.cs_Ha'}")
	else:
	    raise ValueError("Something is wrong in your input file (necessary input file missing)")
        log_list = [cs_H_Log,cs_Ha_Log,cs_N_Log,cs_CA_Log,J_Log,noe_Log]
        for i in range(0,len(log_list)):
            if os.path.isfile(log_list[i]):
                os.system('rm %s'%log_list[i])

        # RMR
# }}}


        # VERY IMPORTANT: compute reference state self.logZ  for the free energies, so they are properly normalized #
        Z = 0.0
        for ensemble_index in range(self.nensembles):
            for s in self.ensembles[ensemble_index]:
                Z +=  np.exp(-s.free_energy)
        self.logZ = np.log(Z)

        # store this constant so we're not recalculating it all the time in neglogP
        self.ln2pi = np.log(2.0*np.pi)
    # }}}

    # Build Alternative Ensembles:{{{
    def build_alternative_ensembles(self):

        ### build binary codes for alternative distance combinations
        nensembles = 2**len(self.ambiguous_groups)
        alt_ensemble_states = []
        for i in range(nensembles):
            binstring = bin(i).replace('0b','')
            while len(binstring) < len(self.ambiguous_groups):
                binstring = '0' + binstring
            alt_ensemble_states.append( [bool(int(binstring[j])) for j in range(len(binstring))] )
        print 'alt_ensemble_states', alt_ensemble_states

        ### for each code, build an ensemble
        print 'Building multiple ensembles with switched distances....'
        for alt_state in alt_ensemble_states[1:]:  # skip the all-False first entry; already have [ ensemble ]
            print '\t', len(self.ambiguous_groups), 'distances with switch-state:', alt_state
            self.ensembles.append( copy.copy(self.ensembles[0]) )
            for j in range(len(self.ambiguous_groups)):
                if alt_state[j]: # if True, switch the distances for this pair of ambiguous groups
                    # we need to switch the distances for all structures in this ensemble
                    for s in self.ensembles[-1]:
                        s.switch_distances(self.ambiguous_groups[j][0], self.ambiguous_groups[j][1])
        print 'self.ensembles', self.ensembles
        self.nensembles = len(self.ensembles)

    # }}}

    # Build Reference Prior NOE (Nuclear Overhauser effect):{{{
    def build_reference_prior_noe(self):	#GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference prior for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference priors (noe) for ensemble', k, 'of', self.nensembles, '...'
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

            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.betas_noe = betas_noe
                s.compute_neglog_reference_priors_noe()
    #}}}

    # Build Gaussian Ref. Prior NOE:{{{
    def build_gaussian_reference_prior_noe(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing gaussian reference priors (noe) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            ndistances = len(ensemble[0].distance_restraints)
#	    print 'ndistances', ndistances
#	    sys.exit()
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
#	    print 'global_ref_sigma_noe ', global_ref_sigma_noe
#	    sys.exit()
#	    l=[5.5,5.5,5.5,5.5,3.5,3.5,2.5,3.5]
#	    m=[1.44,1.77,1.09,1.44,0.81,1.16,0.51,0.88]
#            for j in range(ndistances):
#                ref_sigma_noe[j] = global_ref_sigma_noe
#		ref_sigma_noe[j] = m[j]
#		ref_mean_noe[j] = l[j]
#		print j, ref_sigma_noe[j]
#	    sys.exit()
            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.ref_mean_noe = ref_mean_noe
            	s.ref_sigma_noe = ref_sigma_noe
	        s.compute_gaussian_neglog_reference_priors_noe()
    #}}}

    # Build Ref. Prior H:{{{
    def build_reference_prior_H(self):				#GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference prior for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference priors (H) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_H = len(ensemble[0].chemicalshift_H_restraints)
            all_chemicalshift_H = []
            chemicalshift_H_distributions = [[] for j in range(nchemicalshift_H)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_H_restraints)):
                    chemicalshift_H_distributions[j].append( s.chemicalshift_H_restraints[j].model_chemicalshift_H )
                    all_chemicalshift_H.append( s.chemicalshift_H_restraints[j].model_chemicalshift_H )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_H = np.zeros(nchemicalshift_H)
            for j in range(nchemicalshift_H):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_H[j] =  np.array(chemicalshift_H_distributions[j]).sum()/(len(chemicalshift_H_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.betas_H = betas_H
                s.compute_neglog_reference_priors_H()
    #}}}

    # Build Gaussain Ref H:{{{
    def build_gaussian_reference_prior_H(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference priors (H) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_H = len(ensemble[0].chemicalshift_H_restraints)
            all_chemicalshift_H = []
            chemicalshift_H_distributions = [[] for j in range(nchemicalshift_H)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_H_restraints)):
                    chemicalshift_H_distributions[j].append( s.chemicalshift_H_restraints[j].model_chemicalshift_H )
                    all_chemicalshift_H.append( s.chemicalshift_H_restraints[j].model_chemicalshift_H )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_H = np.zeros(nchemicalshift_H)
	    ref_sigma_H = np.zeros(nchemicalshift_H)
            for j in range(nchemicalshift_H):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_H[j] =  np.array(chemicalshift_H_distributions[j]).mean()
                squared_diffs_H = [ (d - ref_mean_H[j])**2.0 for d in chemicalshift_H_distributions[j] ]
                ref_sigma_H[j] = np.sqrt( np.array(squared_diffs_H).sum() / (len(chemicalshift_H_distributions[j])+1.0))
            global_ref_sigma_H = ( np.array([ref_sigma_H[j]**-2.0 for j in range(nchemicalshift_H)]).mean() )**-0.5
            for j in range(nchemicalshift_H):
                ref_sigma_H[j] = global_ref_sigma_H
#		ref_sigma_H[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.ref_mean_H = ref_mean_H
                s.ref_sigma_H = ref_sigma_H
                s.compute_gaussian_neglog_reference_priors_H()
    #}}}

    # Build Ref Prior Ha:{{{
    def build_reference_prior_Ha(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference prior for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference priors (Ha) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_Ha = len(ensemble[0].chemicalshift_Ha_restraints)
            all_chemicalshift_Ha = []
            chemicalshift_Ha_distributions = [[] for j in range(nchemicalshift_Ha)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_Ha_restraints)):
                    chemicalshift_Ha_distributions[j].append( s.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha )
                    all_chemicalshift_Ha.append( s.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_Ha = np.zeros(nchemicalshift_Ha)
            for j in range(nchemicalshift_Ha):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_Ha[j] =  np.array(chemicalshift_Ha_distributions[j]).sum()/(len(chemicalshift_Ha_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.betas_Ha = betas_Ha
                s.compute_neglog_reference_priors_Ha()
    #}}}

    # Build Gaussian Ref Prior Ha:{{{
    def build_gaussian_reference_prior_Ha(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference priors (Ha) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_Ha = len(ensemble[0].chemicalshift_Ha_restraints)
            all_chemicalshift_Ha = []
            chemicalshift_Ha_distributions = [[] for j in range(nchemicalshift_Ha)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_Ha_restraints)):
                    chemicalshift_Ha_distributions[j].append( s.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha )
                    all_chemicalshift_Ha.append( s.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_Ha = np.zeros(nchemicalshift_Ha)
	    ref_sigma_Ha = np.zeros(nchemicalshift_Ha)
            for j in range(nchemicalshift_Ha):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_Ha[j] =  np.array(chemicalshift_Ha_distributions[j]).mean()
                squared_diffs_Ha = [ (d - ref_mean_Ha[j])**2.0 for d in chemicalshift_Ha_distributions[j] ]
                ref_sigma_Ha[j] = np.sqrt( np.array(squared_diffs_Ha).sum() / (len(chemicalshift_Ha_distributions[j])+1.0))
            global_ref_sigma_Ha = ( np.array([ref_sigma_Ha[j]**-2.0 for j in range(nchemicalshift_Ha)]).mean() )**-0.5
            for j in range(nchemicalshift_Ha):
                ref_sigma_Ha[j] = global_ref_sigma_Ha
#		ref_sigma_Ha[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.ref_mean_Ha = ref_mean_Ha
                s.ref_sigma_Ha = ref_sigma_Ha
                s.compute_gaussian_neglog_reference_priors_Ha()
    #}}}

    # Build Ref Prior N:{{{
    def build_reference_prior_N(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference prior for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference priors (N) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_N = len(ensemble[0].chemicalshift_N_restraints)
            all_chemicalshift_N = []
            chemicalshift_N_distributions = [[] for j in range(nchemicalshift_N)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_N_restraints)):
                    chemicalshift_N_distributions[j].append( s.chemicalshift_N_restraints[j].model_chemicalshift_N )
                    all_chemicalshift_N.append( s.chemicalshift_N_restraints[j].model_chemicalshift_N )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_N = np.zeros(nchemicalshift_N)
            for j in range(nchemicalshift_N):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_N[j] =  np.array(chemicalshift_N_distributions[j]).sum()/(len(chemicalshift_N_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.betas_N = betas_N
                s.compute_neglog_reference_priors_N()

    #:}}}

    # Build Gaussian Ref Prior N:{{{
    def build_gaussian_reference_prior_N(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference priors (N) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_N = len(ensemble[0].chemicalshift_N_restraints)
            all_chemicalshift_N = []
            chemicalshift_N_distributions = [[] for j in range(nchemicalshift_N)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_N_restraints)):
                    chemicalshift_N_distributions[j].append( s.chemicalshift_N_restraints[j].model_chemicalshift_N )
                    all_chemicalshift_N.append( s.chemicalshift_N_restraints[j].model_chemicalshift_N )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_N = np.zeros(nchemicalshift_N)
	    ref_sigma_N = np.zeros(nchemicalshift_N)
            for j in range(nchemicalshift_N):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_N[j] =  np.array(chemicalshift_N_distributions[j]).mean()
                squared_diffs_N = [ (d - ref_mean_N[j])**2.0 for d in chemicalshift_N_distributions[j] ]
#                ref_sigma_N[j] = np.sqrt( np.array(squared_diffs_N).sum() / (len(chemicalshift_N_distributions[j])+1.0))
#            global_ref_sigma_N = ( np.array([ref_sigma_N[j]**-2.0 for j in range(nchemicalshift_N)]).mean() )**-0.5
            for j in range(nchemicalshift_N):
#                ref_sigma_N[j] = global_ref_sigma_N
		ref_sigma_N[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.ref_mean_N = ref_mean_N
                s.ref_sigma_N = ref_sigma_N
                s.compute_gaussian_neglog_reference_priors_N()

    #:}}}

    # Build Ref Prior Ca:{{{
    def build_reference_prior_Ca(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference prior for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference priors (Ca) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_Ca = len(ensemble[0].chemicalshift_Ca_restraints)
            all_chemicalshift_Ca = []
            chemicalshift_Ca_distributions = [[] for j in range(nchemicalshift_Ca)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_Ca_restraints)):
                    chemicalshift_Ca_distributions[j].append( s.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca )
                    all_chemicalshift_Ca.append( s.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_Ca = np.zeros(nchemicalshift_Ca)
            for j in range(nchemicalshift_Ca):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_Ca[j] =  np.array(chemicalshift_Ca_distributions[j]).sum()/(len(chemicalshift_Ca_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.betas_Ca = betas_Ca
                s.compute_neglog_reference_priors_Ca()

    #:}}}

    # Build Gaussian Ref Prior Ca:{{{
    def build_gaussian_reference_prior_Ca(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference priors (Ca) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nchemicalshift_Ca = len(ensemble[0].chemicalshift_Ca_restraints)
            all_chemicalshift_Ca = []
            chemicalshift_Ca_distributions = [[] for j in range(nchemicalshift_Ca)]
            for s in ensemble:
                for j in range(len(s.chemicalshift_Ca_restraints)):
                    chemicalshift_Ca_distributions[j].append( s.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca )
                    all_chemicalshift_Ca.append( s.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca )

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_Ca = np.zeros(nchemicalshift_Ca)
	    ref_sigma_Ca = np.zeros(nchemicalshift_Ca)
            for j in range(nchemicalshift_Ca):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_Ca[j] =  np.array(chemicalshift_Ca_distributions[j]).mean()
                squared_diffs_Ca = [ (d - ref_mean_Ca[j])**2.0 for d in chemicalshift_Ca_distributions[j] ]
#                ref_sigma_Ca[j] = np.sqrt( np.array(squared_diffs_Ca).sum() / (len(chemicalshift_Ca_distributions[j])+1.0))
#            global_ref_sigma_Ca = ( np.array([ref_sigma_Ca[j]**-2.0 for j in range(nchemicalshift_Ca)]).mean() )**-0.5
            for j in range(nchemicalshift_Ca):
#                ref_sigma_Ca[j] = global_ref_sigma_Ca
		ref_sigma_Ca[j] = 12.0
            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.ref_mean_Ca = ref_mean_Ca
                s.ref_sigma_Ca = ref_sigma_Ca
                s.compute_gaussian_neglog_reference_priors_Ca()

    #:}}}

    # Build Ref Prior PF:{{{
    def build_reference_prior_PF(self):                          #GYH
        """Look at all the structures to find the average distances

        >>    beta_j = np.array(distance_distributions[j]).sum()/(len(distance_distributions[j])+1.0)

        then store this info as the reference prior for each structures"""

        for k in range(self.nensembles):

            print 'Computing reference priors (PF) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nprotectionfactor = len(ensemble[0].protectionfactor_restraints)
            all_protectionfactor = []
            protectionfactor_distributions = [[] for j in range(nprotectionfactor)]
            for s in ensemble:
                for j in range(len(s.protectionfactor_restraints)):
                    protectionfactor_distributions[j].append( s.protectionfactor_restraints[j].model_protectionfactor )
                    all_protectionfactor.append( s.protectionfactor_restraints[j].model_protectionfactor )

            # Find the MLE average (i.e. beta_j) for each distance
            betas_PF = np.zeros(nprotectionfactor)
            for j in range(nprotectionfactor):
                # plot the maximum likelihood exponential distribution fitting the data
                betas_PF[j] =  np.array(protectionfactor_distributions[j]).sum()/(len(protectionfactor_distributions[j])+1.0)

            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.betas_PF = betas_PF
                s.compute_neglog_reference_priors_PF()

    #:}}}

    # Build Gaussian Ref Prior PF:{{{
    def build_gaussian_reference_prior_PF(self):        #GYH

        for k in range(self.nensembles):

            print 'Computing Gaussian reference priors (PF) for ensemble', k, 'of', self.nensembles, '...'
            ensemble = self.ensembles[k]

            # collect distance distributions across all structures
            nprotectionfactor = len(ensemble[0].protectionfactor_restraints)
            all_protectionfactor = []
            protectionfactor_distributions = [[] for j in range(nprotectionfactor)]
            for s in ensemble:
                for j in range(len(s.protectionfactor_restraints)):
                    protectionfactor_distributions[j].append( s.protectionfactor_restraints[j].model_protectionfactor )
                    all_protectionfactor.append( s.protectionfactor_restraints[j].model_protectionfactor )
#	    print 'all_protectionfactor', all_protectionfactor
	    print 'len(all_protectionfactor)', len(all_protectionfactor)

            # Find the MLE average (i.e. beta_j) for each distance
            ref_mean_PF = np.zeros(nprotectionfactor)
	    ref_sigma_PF = np.zeros(nprotectionfactor)
#	    squared_diffs_PF = []
            for j in range(nprotectionfactor):
                # plot the maximum likelihood exponential distribution fitting the data
                ref_mean_PF[j] =  np.array(protectionfactor_distributions[j]).mean()
#                squared_diffs_PF.append( [ (d - ref_mean_PF[j])**2.0 for d in protectionfactor_distributions[j] ])
	    	squared_diffs_PF=( [ (d - ref_mean_PF[j])**2.0 for d in protectionfactor_distributions[j] ])
#		ref_sigma_PF[j] = np.sqrt( np.array(squared_diffs_PF).sum() / (len(protectionfactor_distributions[j])+1.0))
#            global_ref_sigma_PF = ( np.array([ref_sigma_PF[j]**-2.0 for j in range(nprotectionfactor)]).mean() )**-0.5

#            global_ref_sigma_PF = np.array(all_protectionfactor).std()
#            print 'global_ref_sigma_PF', global_ref_sigma_PF
#            sys.exit(1)

#np.sqrt( np.array(squared_diffs_PF).sum() / (len(squared_diffs_PF) + nprotectionfactor))
            for j in range(nprotectionfactor):
#                ref_sigma_PF[j] = global_ref_sigma_PF #np.sqrt( np.array(squared_diffs_PF).sum() / (len(protectionfactor_distributions[j])+1.0))
		ref_sigma_PF[j] = 20.0
#		ref_sigma_PF[j] = 1.84394160179	#np.sqrt( np.array(squared_diffs_PF).sum() / (len(protectionfactor_distributions[j])+1.0))
            # store the beta information in each structure and compute/store the -log P_prior
            for s in ensemble:
                s.ref_mean_PF = ref_mean_PF
                s.ref_sigma_PF =  ref_sigma_PF
                s.compute_gaussian_neglog_reference_priors_PF()

    #:}}}

    # -log(P):{{{
    def neglogP(self, new_ensemble_index, new_state, new_sigma_noe, new_sigma_J, new_sigma_cs_H, new_sigma_cs_Ha, new_sigma_cs_N, new_sigma_cs_Ca, new_sigma_PF, new_gamma_index, new_alpha_index, verbose=True):	#GYH
        """Return -ln P of the current configuration."""

        # The current structure being sampled
        s = self.ensembles[new_ensemble_index][new_state]

        # model terms
        result = s.free_energy  + self.logZ

        # distance terms
        #result += (Nj+1.0)*np.log(self.sigma_noe)
       	if s.sse_distances is not None:		# trying to fix a future warning:"comparison to `None` will result in an elementwise object comparison in the future."
		result += (s.Ndof_distances+3.0)*np.log(new_sigma_noe)  # for use with log-spaced sigma values
		result += s.sse_distances[new_gamma_index] / (2.0*new_sigma_noe**2.0)
        	result += (s.Ndof_distances)/2.0*self.ln2pi  # for normalization

        # dihedral terms
        if s.sse_dihedrals != None:
		result += (s.Ndof_dihedrals)*np.log(new_sigma_J) # for use with log-spaced sigma values
		result += s.sse_dihedrals / (2.0*new_sigma_J**2.0)
        	result += (s.Ndof_dihedrals)/2.0*self.ln2pi  # for normalization

	# chemicalshift terms                           # GYH
	if s.sse_chemicalshift_H != None:
       		result += (s.Ndof_chemicalshift_H)*np.log(new_sigma_cs_H)
         	result += s.sse_chemicalshift_H / (2.0*new_sigma_cs_H**2.0)
        	result += (s.Ndof_chemicalshift_H)/2.0*self.ln2pi
	if s.sse_chemicalshift_Ha != None:
		result += (s.Ndof_chemicalshift_Ha)*np.log(new_sigma_cs_Ha)
	        result += s.sse_chemicalshift_Ha / (2.0*new_sigma_cs_Ha**2.0)
	        result += (s.Ndof_chemicalshift_Ha)/2.0*self.ln2pi
	if s.sse_chemicalshift_N != None:
	        result += (s.Ndof_chemicalshift_N)*np.log(new_sigma_cs_N)
	        result += s.sse_chemicalshift_N / (2.0*new_sigma_cs_N**2.0)
	        result += (s.Ndof_chemicalshift_N)/2.0*self.ln2pi
	if s.sse_chemicalshift_Ca != None:
	        result += (s.Ndof_chemicalshift_Ca)*np.log(new_sigma_cs_Ca)
	        result += s.sse_chemicalshift_Ca / (2.0*new_sigma_cs_Ca**2.0)
	        result += (s.Ndof_chemicalshift_Ca)/2.0*self.ln2pi


        # protectionfactor terms                           # GYH
	if s.sse_protectionfactor is not None:			# trying to fix a future warning:"comparison to `None` will result in an elementwise object comparison in the future."
	        result += (s.Ndof_protectionfactor)*np.log(new_sigma_PF)
	        result += s.sse_protectionfactor[new_alpha_index] / (2.0*new_sigma_PF**2.0)
	        result += (s.Ndof_protectionfactor)/2.0*self.ln2pi


        # reference priors
        if self.use_reference_prior_noe:
            result -= s.sum_neglog_reference_priors_noe		#GYH
        if self.use_reference_prior_H:
            result -= s.sum_neglog_reference_priors_H		#GYH
        if self.use_reference_prior_Ha:
            result -= s.sum_neglog_reference_priors_Ha		#GYH
        if self.use_reference_prior_N:
            result -= s.sum_neglog_reference_priors_N		#GYH
        if self.use_reference_prior_Ca:
            result -= s.sum_neglog_reference_priors_Ca		#GYH
        if self.use_reference_prior_PF:
            result -= s.sum_neglog_reference_priors_PF		#GYH

        if self.use_gaussian_reference_prior_noe:
            result -= s.sum_gaussian_neglog_reference_priors_noe         #GYH
        if self.use_gaussian_reference_prior_H:
            result -= s.sum_gaussian_neglog_reference_priors_H           #GYH
        if self.use_gaussian_reference_prior_Ha:
            result -= s.sum_gaussian_neglog_reference_priors_Ha          #GYH
        if self.use_gaussian_reference_prior_N:
            result -= s.sum_gaussian_neglog_reference_priors_N           #GYH
        if self.use_gaussian_reference_prior_Ca:
            result -= s.sum_gaussian_neglog_reference_priors_Ca          #GYH
        if self.use_gaussian_reference_prior_PF:
            result -= s.sum_gaussian_neglog_reference_priors_PF          #GYH


        if verbose:
            print 'state, f_sim', new_state, s.free_energy,
            print 's.sse_distances[', new_gamma_index, ']', s.sse_distances[new_gamma_index], 's.Ndof_distances', s.Ndof_distances
            print 's.sse_dihedrals', s.sse_dihedrals, 's.Ndof_dihedrals', s.Ndof_dihedrals
	    print 's.sse_chemicalshift_H', s.sse_chemicalshift_H, 's.Ndof_chemicalshift_H', s.Ndof_chemicalshift_H # GYH
            print 's.sse_chemicalshift_Ha', s.sse_chemicalshift_Ha, 's.Ndof_chemicalshift_Ha', s.Ndof_chemicalshift_Ha # GYH
            print 's.sse_chemicalshift_N', s.sse_chemicalshift_N, 's.Ndof_chemicalshift_N', s.Ndof_chemicalshift_N # GYH
            print 's.sse_chemicalshift_Ca', s.sse_chemicalshift_Ca, 's.Ndof_chemicalshift_Ca', s.Ndof_chemicalshift_Ca # GYH
	    print 's.sse_protectionfactor[', new_alpha_index, ']', s.sse_protectionfactor[new_alpha_index], 's.Ndof_protectionfactor', s.Ndof_protectionfactor #GYH
	    print 's.sum_neglog_reference_priors_noe', s.sum_neglog_reference_priors_noe, 's.sum_neglog_reference_priors_H', s.sum_neglog_reference_priors_H, 's.sum_neglog_reference_priors_Ha',s.sum_neglog_reference_priors_Ha, 's.sum_neglog_reference_priors_N', s.sum_neglog_reference_priors_N, 's.sum_neglog_reference_priors_Ca', s.sum_neglog_reference_priors_Ca, 's.sum_neglog_reference_priors_PF', s.sum_neglog_reference_priors_PF	#GYH
	    print 's.sum_gaussian_neglog_reference_priors_noe', s.sum_gaussian_neglog_reference_priors_noe, 's.sum_gaussian_neglog_reference_priors_H', s.sum_gaussian_neglog_reference_priors_H, 's.sum_gaussian_neglog_reference_priors_Ha', s.sum_gaussian_neglog_reference_priors_Ha, 's.sum_gaussian_neglog_reference_priors_N', s.sum_gaussian_neglog_reference_priors_N, 's.sum_gaussian_neglog_reference_priors_Ca', s.sum_gaussian_neglog_reference_priors_Ca, 's.sum_gaussian_neglog_reference_priors_PF', s.sum_gaussian_neglog_reference_priors_PF       #GYH
        return result
    # }}}

## Compute -log( reference priors (ALL Restraints) ):{{{
#    def compute_neglog_reference_priors_noe(self):		#GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print 'self.betas', self.betas
#
#        self.neglog_reference_priors_noe = np.zeros(self.ndistances)
#        self.sum_neglog_reference_priors_noe = 0.
#        for j in range(self.ndistances):
#            self.neglog_reference_priors_noe[j] = np.log(self.betas_noe[j]) + self.distance_restraints[j].model_distance/self.betas_noe[j]
#            self.sum_neglog_reference_priors_noe  += self.distance_restraints[j].weight * self.neglog_reference_priors_noe[j]
#
#    def compute_gaussian_neglog_reference_priors_noe(self):	#GYH
#	"""An alternative option for reference potential based on Gaussian distribution"""
#	self.gaussian_neglog_reference_priors_noe = np.zeros(self.ndistances)
#	self.sum_gaussian_neglog_reference_priors_noe = 0.
#	for j in range(self.ndistances):
#	    self.gaussian_neglog_reference_priors_noe[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_noe[j]) + (self.distance_restraints[j].model_distance - self.ref_mean_noe[j])**2.0/(2*(self.ref_sigma_noe[j]**2.0))
#	    self.sum_gaussian_neglog_reference_priors_noe += self.distance_restraints[j].weight * self.gaussian_neglog_reference_priors_noe[j]
#
#    def compute_neglog_reference_priors_H(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print 'self.betas', self.betas
#
#        self.neglog_reference_priors_H = np.zeros(self.nchemicalshift_H)
#        self.sum_neglog_reference_priors_H = 0.
#        for j in range(self.nchemicalshift_H):
#            self.neglog_reference_priors_H[j] = np.log(self.betas_H[j]) + self.chemicalshift_H_restraints[j].model_chemicalshift_H/self.betas_H[j]
#            self.sum_neglog_reference_priors_H  += self.chemicalshift_H_restraints[j].weight * self.neglog_reference_priors_H[j]
#
#    def compute_gaussian_neglog_reference_priors_H(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_priors_H = np.zeros(self.nchemicalshift_H)
#        self.sum_gaussian_neglog_reference_priors_H = 0.
#        for j in range(self.nchemicalshift_H):
#            self.gaussian_neglog_reference_priors_H[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_H[j]) + (self.chemicalshift_H_restraints[j].model_chemicalshift_H - self.ref_mean_H[j])**2.0/(2*self.ref_sigma_H[j]**2.0)
#            self.sum_gaussian_neglog_reference_priors_H += self.chemicalshift_H_restraints[j].weight * self.gaussian_neglog_reference_priors_H[j]
#
#
#    def compute_neglog_reference_priors_Ha(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print 'self.betas', self.betas
#
#        self.neglog_reference_priors_Ha = np.zeros(self.nchemicalshift_Ha)
#        self.sum_neglog_reference_priors_Ha = 0.
#        for j in range(self.nchemicalshift_Ha):
#            self.neglog_reference_priors_Ha[j] = np.log(self.betas_Ha[j]) + self.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha/self.betas_Ha[j]
#            self.sum_neglog_reference_priors_Ha  += self.chemicalshift_Ha_restraints[j].weight * self.neglog_reference_priors_Ha[j]
#
#    def compute_gaussian_neglog_reference_priors_Ha(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_priors_Ha = np.zeros(self.nchemicalshift_Ha)
#        self.sum_gaussian_neglog_reference_priors_Ha = 0.
#        for j in range(self.nchemicalshift_Ha):
#            self.gaussian_neglog_reference_priors_Ha[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ha[j]) + (self.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha - self.ref_mean_Ha[j])**2.0/(2*self.ref_sigma_Ha[j]**2.0)
#            self.sum_gaussian_neglog_reference_priors_Ha += self.chemicalshift_Ha_restraints[j].weight * self.gaussian_neglog_reference_priors_Ha[j]
#
#    def compute_neglog_reference_priors_N(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print 'self.betas', self.betas
#
#        self.neglog_reference_priors_N = np.zeros(self.nchemicalshift_N)
#        self.sum_neglog_reference_priors_N = 0.
#        for j in range(self.nchemicalshift_N):
#            self.neglog_reference_priors_N[j] = np.log(self.betas_N[j]) + self.chemicalshift_N_restraints[j].model_chemicalshift_N/self.betas_N[j]
#            self.sum_neglog_reference_priors_N  += self.chemicalshift_N_restraints[j].weight * self.neglog_reference_priors_N[j]
#
#    def compute_gaussian_neglog_reference_priors_N(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_priors_N = np.zeros(self.nchemicalshift_N)
#        self.sum_gaussian_neglog_reference_priors_N = 0.
#        for j in range(self.nchemicalshift_N):
#            self.gaussian_neglog_reference_priors_N[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_N[j]) + (self.chemicalshift_N_restraints[j].model_chemicalshift_N - self.ref_mean_N[j])**2.0/(2*self.ref_sigma_N[j]**2.0)
#            self.sum_gaussian_neglog_reference_priors_N += self.chemicalshift_N_restraints[j].weight * self.gaussian_neglog_reference_priors_N[j]
#
#
#    def compute_neglog_reference_priors_Ca(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print 'self.betas', self.betas
#
#        self.neglog_reference_priors_Ca = np.zeros(self.nchemicalshift_Ca)
#        self.sum_neglog_reference_priors_Ca = 0.
#        for j in range(self.nchemicalshift_Ca):
#            self.neglog_reference_priors_Ca[j] = np.log(self.betas_Ca[j]) + self.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca/self.betas_Ca[j]
#            self.sum_neglog_reference_priors_Ca  += self.chemicalshift_Ca_restraints[j].weight * self.neglog_reference_priors_Ca[j]
#
#    def compute_gaussian_neglog_reference_priors_Ca(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_priors_Ca = np.zeros(self.nchemicalshift_Ca)
#        self.sum_gaussian_neglog_reference_priors_Ca = 0.
#        for j in range(self.nchemicalshift_Ca):
#            self.gaussian_neglog_reference_priors_Ca[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ca[j]) + (self.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
#            self.sum_gaussian_neglog_reference_priors_Ca += self.chemicalshift_Ca_restraints[j].weight * self.gaussian_neglog_reference_priors_Ca[j]
#
#
#    def compute_neglog_reference_priors_PF(self):              #GYH
#        """Uses the stored beta information (calculated across all structures) to calculate
#        - log P_ref(distance[j) for each distance j."""
#
#        # print 'self.betas', self.betas
#
#        self.neglog_reference_priors_PF= np.zeros(self.nprotectionfactor)
#        self.sum_neglog_reference_priors_PF = 0.
#        for j in range(self.nprotectionfactor):
#            self.neglog_reference_priors_PF[j] = np.log(self.betas_PF[j]) + self.protectionfactor_restraints[j].model_protectionfactor/self.betas_PF[j]
#            self.sum_neglog_reference_priors_PF  += self.protectionfactor_restraints[j].weight * self.neglog_reference_priors_PF[j]
#
#
#    def compute_gaussian_neglog_reference_priors_PF(self):     #GYH
#        """An alternative option for reference potential based on Gaussian distribution"""
#        self.gaussian_neglog_reference_priors_PF = np.zeros(self.nprotectionfactor)
#        self.sum_gaussian_neglog_reference_priors_PF = 0.
#        for j in range(self.nprotectionfactor):
##	    print j, 'self.ref_sigma_PF[j]', self.ref_sigma_PF[j], 'self.ref_mean_PF[j]', self.ref_mean_PF[j]
#            self.gaussian_neglog_reference_priors_PF[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_PF[j]) + (self.protectionfactor_restraints[j].model_protectionfactor - self.ref_mean_PF[j])**2.0/(2*self.ref_sigma_PF[j]**2.0)
#            self.sum_gaussian_neglog_reference_priors_PF += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_priors_PF[j]
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
            new_sigma_PF = self.sigma_PF                #GYH
            new_sigma_PF_index = self.sigma_PF_index    #GYH
	    new_gamma = self.gamma
            new_gamma_index = self.gamma_index
	    new_alpha = self.alpha	#GYH
	    new_alpha_index = self.alpha_index	#GYH

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




	    elif np.random.random() < 0.60 :	#GYH
		# take a step in array of allowed sigma_PF
                new_sigma_PF_index += (np.random.randint(3)-1)
                new_sigma_PF_index = new_sigma_PF_index%(len(self.allowed_sigma_PF)) # don't go out of bounds
                new_sigma_PF = self.allowed_sigma_PF[new_sigma_PF_index]


            elif np.random.random() < 0.78:
                # take a step in array of allowed gamma
                new_gamma_index += (np.random.randint(3)-1)
                new_gamma_index = new_gamma_index%(len(self.allowed_gamma)) # don't go out of bounds
                new_gamma  = self.allowed_gamma[new_gamma_index]


	    elif np.random.random() < 0.88:	#GYH
                # take a step in array of allowed alpha
                new_alpha_index += (np.random.randint(3)-1)
                new_alpha_index = new_alpha_index%(len(self.allowed_alpha)) # don't go out of bounds
                new_alpha  = self.allowed_alpha[new_alpha_index]


            elif np.random.random() < 0.99:
                # take a random step in state space
                new_state = np.random.randint(self.nstates)

            else:
                # pick a random pair of ambiguous groups to switch
                new_ensemble_index = np.random.randint(self.nensembles)

            # compute new "energy"
            verbose = False
#            if step%self.print_every == 0:
#                verbose = True
            new_E = self.neglogP(new_ensemble_index, new_state, new_sigma_noe, new_sigma_J, new_sigma_cs_H, new_sigma_cs_Ha, new_sigma_cs_N, new_sigma_cs_Ca, new_sigma_PF, new_gamma_index, new_alpha_index,  verbose=verbose)

            # accept or reject the MC move according to Metroplis criterion
            accept = False
            if new_E < self.E:
                accept = True
            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

            # print status
#            if step%self.print_every == 0:
#                print 'step', step, 'E', self.E, 'new_E', new_E, 'accept', accept, 'new_sigma_noe', new_sigma_noe, 'new_sigma_J', new_sigma_J, 'new_sigma_cs_H', new_sigma_cs_H, 'new_sigma_cs_Ha', new_sigma_cs_Ha, 'new_sigma_cs_N', new_sigma_cs_N, 'new_sigma_cs_Ca', new_sigma_cs_Ca, 'new_sigma_PF', new_sigma_PF, 'new_gamma', new_gamma, 'new_alpha', new_alpha, 'new_state', new_state, 'new_ensemble_index', new_ensemble_index
	    # Store trajectory counts
            self.traj.sampled_sigma_noe[self.sigma_noe_index] += 1
            self.traj.sampled_sigma_J[self.sigma_J_index] += 1
	    self.traj.sampled_sigma_cs_H[self.sigma_cs_H_index] += 1  #GYH
            self.traj.sampled_sigma_cs_Ha[self.sigma_cs_Ha_index] += 1  #GYH
            self.traj.sampled_sigma_cs_N[self.sigma_cs_N_index] += 1  #GYH
            self.traj.sampled_sigma_cs_Ca[self.sigma_cs_Ca_index] += 1  #GYH
	    self.traj.sampled_sigma_PF[self.sigma_PF_index] += 1 #GYH
            self.traj.sampled_gamma[self.gamma_index] += 1
	    self.traj.sampled_alpha[self.alpha_index] += 1	#GYH
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
                self.sigma_PF = new_sigma_PF                    #GYH
                self.sigma_PF_index = new_sigma_PF_index        #GYH
		self.gamma = new_gamma
                self.gamma_index = new_gamma_index
		self.alpha = new_alpha	#GYH
		self.alpha_index = new_alpha_index	#GYH
                self.state = new_state
                self.ensemble_index = new_ensemble_index
                self.accepted += 1.0
            self.total += 1.0

            # store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step), float(self.E), int(accept), int(self.state), int(self.sigma_noe_index), int(self.sigma_J_index), int(self.sigma_cs_H_index), int(self.sigma_cs_Ha_index), int(self.sigma_cs_N_index), int(self.sigma_cs_Ca_index), int(self.sigma_PF_index), int(self.gamma_index), int(self.alpha_index)] )	#GYH

#            if step%self.print_every == 0:
#                print 'accratio =', self.accepted/self.total
    #}}}

# }}}

# Class PosteriorSamplingTrajectory:{{{
class PosteriorSamplingTrajectory(object):
    "A class to store and perform operations on the trajectories of sampling runs."

    # __init__:{{{
    def __init__(self, ensemble, allowed_sigma_noe, allowed_sigma_J,
            allowed_sigma_cs_H, allowed_sigma_cs_Ha, allowed_sigma_cs_N,
            allowed_sigma_cs_Ca, allowed_sigma_PF, allowed_gamma, allowed_alpha):	#GYH
        "Initialize the PosteriorSamplingTrajectory."

        self.nstates = len(ensemble)
        self.ensemble = ensemble
        self.ndistances = len(self.ensemble[0].distance_restraints)
        self.ndihedrals = len(self.ensemble[0].dihedral_restraints)
	self.nchemicalshift_H = len(self.ensemble[0].chemicalshift_H_restraints) #GYH
        self.nchemicalshift_Ha = len(self.ensemble[0].chemicalshift_Ha_restraints) #GYH
        self.nchemicalshift_Ca = len(self.ensemble[0].chemicalshift_Ca_restraints) #GYH
        self.nchemicalshift_N = len(self.ensemble[0].chemicalshift_N_restraints) #GYH
	self.nprotectionfactor = len(self.ensemble[0].protectionfactor_restraints) #GYH

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

        self.allowed_sigma_PF = allowed_sigma_PF                        #GYH
        self.sampled_sigma_PF = np.zeros(len(allowed_sigma_PF))         #GYH

        self.allowed_gamma = allowed_gamma
        self.sampled_gamma = np.zeros(len(allowed_gamma))

	self.allowed_alpha = allowed_alpha	#GYH
	self.sampled_alpha = np.zeros(len(allowed_alpha))	#GYH

        self.state_counts = np.ones(self.nstates)  # add a pseudocount to avoid log(0) errors

        self.f_sim = np.array([e.free_energy for e in ensemble])
        self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()

        # stores samples [step, self.E, accept, state, sigma_noe, sigma_J, sigma_cs, gamma]
        self.trajectory_headers = ['step', 'E', 'accept', 'state', 'sigma_noe_index', 'sigma_J_index', 'sigma_cs_H_index', 'sigma_cs_Ha_index', 'sigma_cs_N_index', 'sigma_cs_Ca_index', 'sigma_PF_index', 'gamma_index', 'alpha_index']	#GYH
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
	self.results['allowed_sigma_PF'] = self.allowed_sigma_PF.tolist() #GYH
        self.results['allowed_gamma'] = self.allowed_gamma.tolist()
 	self.results['allowed_alpha'] = self.allowed_alpha.tolist()	#GYH
        self.results['sampled_sigma_noe'] = self.sampled_sigma_noe.tolist()
        self.results['sampled_sigma_J'] = self.sampled_sigma_J.tolist()
        self.results['sampled_sigma_cs_H'] = self.sampled_sigma_cs_H.tolist()   #GYH
        self.results['sampled_sigma_cs_Ha'] = self.sampled_sigma_cs_Ha.tolist()   #GYH
        self.results['sampled_sigma_cs_N'] = self.sampled_sigma_cs_N.tolist()   #GYH
        self.results['sampled_sigma_cs_Ca'] = self.sampled_sigma_cs_Ca.tolist()   #GYH
	self.results['sampled_sigma_PF'] = self.sampled_sigma_PF.tolist()   #GYH
        self.results['sampled_gamma'] = self.sampled_gamma.tolist()
	self.results['sampled_alpha'] = self.sampled_alpha.tolist()	#GYH

        # Calculate the modes of the nuisance parameter marginal distributions
        self.results['sigma_noe_mode'] = float(self.allowed_sigma_noe[ np.argmax(self.sampled_sigma_noe) ])
        self.results['sigma_J_mode']   = float(self.allowed_sigma_J[ np.argmax(self.sampled_sigma_J) ])
        self.results['sigma_cs_H_mode']   = float(self.allowed_sigma_cs_H[ np.argmax(self.sampled_sigma_cs_H) ])      #GYH
        self.results['sigma_cs_Ha_mode']   = float(self.allowed_sigma_cs_Ha[ np.argmax(self.sampled_sigma_cs_Ha) ])      #GYH
        self.results['sigma_cs_N_mode']   = float(self.allowed_sigma_cs_N[ np.argmax(self.sampled_sigma_cs_N) ])      #GYH
        self.results['sigma_cs_Ca_mode']   = float(self.allowed_sigma_cs_Ca[ np.argmax(self.sampled_sigma_cs_Ca) ])      #GYH
	self.results['sigma_PF_mode']	= float(self.allowed_sigma_PF[ np.argmax(self.sampled_sigma_PF) ])	#GYH
        self.results['gamma_mode']     = float(self.allowed_gamma[ np.argmax(self.sampled_gamma) ])
	self.results['alpha_mode']	= float(self.allowed_alpha[ np.argmax(self.sampled_alpha) ])	#GYH

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
        mean_Jcoupling = (mean_Jcoupling/Z)	#GYH
        self.results['mean_Jcoupling'] = mean_Jcoupling.tolist()

        # Compute the experiment Jcouplings
        exp_Jcoupling = np.array([self.ensemble[0].dihedral_restraints[j].exp_Jcoupling for j in range(self.ndihedrals)])
        self.results['exp_Jcoupling'] = exp_Jcoupling.tolist()
        abs_Jdiffs = np.abs( exp_Jcoupling - mean_Jcoupling )
        self.results['disagreement_Jcoupling_mean'] = float(abs_Jdiffs.mean())
        self.results['disagreement_Jcoupling_std'] = float(abs_Jdiffs.std())

        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_chemicalshift_H = np.zeros(self.nchemicalshift_H)
        Z = np.zeros(self.nchemicalshift_H)
        for i in range(self.nstates):
            for j in range(self.nchemicalshift_H):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].chemicalshift_H_restraints[j].weight
                r = self.ensemble[i].chemicalshift_H_restraints[j].model_chemicalshift_H
                mean_chemicalshift_H[j] += pop*weight*r
                Z[j] += pop*weight
        mean_chemicalshift_H = (mean_chemicalshift_H/Z)
        self.results['mean_chemicalshift_H'] = mean_chemicalshift_H.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_chemicalshift_H = np.array([self.ensemble[0].chemicalshift_H_restraints[j].exp_chemicalshift_H for j in range(self.nchemicalshift_H)])
        self.results['exp_chemicalshift_H'] = exp_chemicalshift_H.tolist()
        abs_cs_H_diffs = np.abs( exp_chemicalshift_H - mean_chemicalshift_H )
        self.results['disagreement_chemicalshift_H_mean'] = float(abs_cs_H_diffs.mean())
        self.results['disagreement_chemicalshift_H_std'] = float(abs_cs_H_diffs.std())

        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_chemicalshift_Ha = np.zeros(self.nchemicalshift_Ha)
        Z = np.zeros(self.nchemicalshift_Ha)
        for i in range(self.nstates):
            for j in range(self.nchemicalshift_Ha):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].chemicalshift_Ha_restraints[j].weight
                r = self.ensemble[i].chemicalshift_Ha_restraints[j].model_chemicalshift_Ha
                mean_chemicalshift_Ha[j] += pop*weight*r
                Z[j] += pop*weight
        mean_chemicalshift_Ha = (mean_chemicalshift_Ha/Z)
        self.results['mean_chemicalshift_Ha'] = mean_chemicalshift_Ha.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_chemicalshift_Ha = np.array([self.ensemble[0].chemicalshift_Ha_restraints[j].exp_chemicalshift_Ha for j in range(self.nchemicalshift_Ha)])
        self.results['exp_chemicalshift_Ha'] = exp_chemicalshift_Ha.tolist()
        abs_cs_Ha_diffs = np.abs( exp_chemicalshift_Ha - mean_chemicalshift_Ha )
        self.results['disagreement_chemicalshift_Ha_mean'] = float(abs_cs_Ha_diffs.mean())
        self.results['disagreement_chemicalshift_Ha_std'] = float(abs_cs_Ha_diffs.std())


        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_chemicalshift_N = np.zeros(self.nchemicalshift_N)
        Z = np.zeros(self.nchemicalshift_N)
        for i in range(self.nstates):
            for j in range(self.nchemicalshift_N):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].chemicalshift_N_restraints[j].weight
                r = self.ensemble[i].chemicalshift_N_restraints[j].model_chemicalshift_N
                mean_chemicalshift_N[j] += pop*weight*r
                Z[j] += pop*weight
        mean_chemicalshift_N = (mean_chemicalshift_N/Z)
        self.results['mean_chemicalshift_N'] = mean_chemicalshift_N.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_chemicalshift_N = np.array([self.ensemble[0].chemicalshift_N_restraints[j].exp_chemicalshift_N for j in range(self.nchemicalshift_N)])
        self.results['exp_chemicalshift_N'] = exp_chemicalshift_N.tolist()
        abs_cs_N_diffs = np.abs( exp_chemicalshift_N - mean_chemicalshift_N )
        self.results['disagreement_chemicalshift_N_mean'] = float(abs_cs_N_diffs.mean())
        self.results['disagreement_chemicalshift_N_std'] = float(abs_cs_N_diffs.std())


        # Estimate the ensemble-averaged chemical shift values    #GYH
        mean_chemicalshift_Ca = np.zeros(self.nchemicalshift_Ca)
        Z = np.zeros(self.nchemicalshift_Ca)
        for i in range(self.nstates):
            for j in range(self.nchemicalshift_Ca):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].chemicalshift_Ca_restraints[j].weight
                r = self.ensemble[i].chemicalshift_Ca_restraints[j].model_chemicalshift_Ca
                mean_chemicalshift_Ca[j] += pop*weight*r
                Z[j] += pop*weight
        mean_chemicalshift_Ca = (mean_chemicalshift_Ca/Z)
        self.results['mean_chemicalshift_Ca'] = mean_chemicalshift_Ca.tolist()

        # Compute the experiment chemical shift                 #GYH
        exp_chemicalshift_Ca = np.array([self.ensemble[0].chemicalshift_Ca_restraints[j].exp_chemicalshift_Ca for j in range(self.nchemicalshift_Ca)])
        self.results['exp_chemicalshift_Ca'] = exp_chemicalshift_Ca.tolist()
        abs_cs_Ca_diffs = np.abs( exp_chemicalshift_Ca - mean_chemicalshift_Ca )
        self.results['disagreement_chemicalshift_Ca_mean'] = float(abs_cs_Ca_diffs.mean())
        self.results['disagreement_chemicalshift_Ca_std'] = float(abs_cs_Ca_diffs.std())



        # Estimate the ensemble-averaged protection factor values    #GYH
        mean_protectionfactor = np.zeros(self.nprotectionfactor)
        Z = np.zeros(self.nprotectionfactor)
        for i in range(self.nstates):
            for j in range(self.nprotectionfactor):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[i].protectionfactor_restraints[j].weight
                r = self.ensemble[i].protectionfactor_restraints[j].model_protectionfactor
                mean_protectionfactor[j] += pop*weight*r
                Z[j] += pop*weight
        mean_protectionfactor = (mean_protectionfactor/Z)	#GYH
        self.results['mean_protectionfactor'] = mean_protectionfactor.tolist()

        # Compute the experiment protection factor                 #GYH

	exp_protectionfactor = np.array([self.ensemble[0].protectionfactor_restraints[j].exp_protectionfactor + self.results['alpha_mode'] for j in range(self.nprotectionfactor)])
#        exp_protectionfactor = np.array([self.ensemble[0].protectionfactor_restraints[j].exp_protectionfactor for j in range(self.nprotectionfactor)])
        self.results['exp_protectionfactor'] = exp_protectionfactor.tolist()
        abs_PFdiffs = np.abs( exp_protectionfactor - mean_protectionfactor )
        self.results['disagreement_protectionfactor_mean'] = float(abs_PFdiffs.mean())
        self.results['disagreement_protectionfactor_std'] = float(abs_PFdiffs.std())
    #}}}

    # log Spaced Array:{{{
    def logspaced_array(self, xmin, xmax, nsteps):
        ymin, ymax = np.log(xmin), np.log(xmax)
        dy = (ymax-ymin)/nsteps
        return np.exp(np.arange(ymin, ymax, dy))
    #}}}

    # Write Results:{{{
    def write_results(self, outfilename='traj.yaml'):
        """Dumps results to a YAML format file. """

        # Read in the YAML data as a dictionary
        fout = file(outfilename, 'w')
        yaml.dump(self.results, fout, default_flow_style=False)
    #}}}


#}}}





