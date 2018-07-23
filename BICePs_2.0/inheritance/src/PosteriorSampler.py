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
#cimport numpy as np
#import cython
from scipy  import loadtxt, savetxt
from matplotlib import pylab as plt
from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from Restraint import *   # Import the Restraint Parent Class as R
from toolbox import *

##############################################################################
# Main
##############################################################################

class PosteriorSampler(object):
    """A class to perform posterior sampling of conformational populations

    INPUTS

        ensemble        - a list of lists of Restraint objects, one list for each conformation.

    OPTIONS

        freq_write_traj - the frequency (in steps) to write the MCMC trajectory
        freq_print      - the frequency (in steps) to print status
        freq_save_traj  - the frequency (in steps) to store the MCMC trajectory

    HISTORY of changes

       July 20, 2018 - changed ensembles[0] to ensemble - NO multiple ensembles!(VAV)
                     - removed the reference potential info -- these are in each child Restaint() class (VAV)

    """

    def __init__(self, ensemble, freq_write_traj=1000, freq_print=1000, freq_save_traj=100):

        """Initialize PosteriorSampler Class."""

        # Allow the ensemble to pass through the class
        self.ensemble = ensemble

        # Step frequencies to write trajectory info
        self.write_traj = freq_write_traj

        # Frequency of printing to the screen
        self.print_every = freq_print # debug

        # Frequency of storing trajectory samples
        self.traj_every = freq_save_traj

        # Ensemble is a list of Restraint objects
        self.nstates = len(ensemble)

        # the initial state of the structural ensemble we're sampling from
        self.state = 0    # index in the ensemble
        self.E = 1.0e99   # initial energy
        self.accepted = 0
        self.total = 0

        # keep track of what we sampled in a trajectory
#NOTE        self.traj = PosteriorSamplingTrajectory(self.ensemble)  # VAV not needed:, self.allowed_sigma, self.allowed_gamma)

        # Go through each restraint type, and construct the specified reference potential if needed

        ## the list of Restraints should be the same for all structures in the ensemble -
	## ... use the first structure's list to determine what kind of reference potential each Restraint has
        ref_types = [ R.ref for R in ensemble[0] ]
        #for R in ensemble[0]:
        #    ref_types = R.ref

        # for each Restraint, calculate global reference potential parameters by looking across all structures
        for rest_index in range(len(ensemble[0])):

            if ref_types[rest_index] == 'uniform':
                pass
            elif ref_types[rest_index] == 'exp':
                self.build_exp_ref(rest_index)
            elif ref_types[rest_index] == 'gaussian':
                self.build_gaussian_ref(rest_index,
                        use_global_ref_sigma=self.ensemble[0][rest_index].use_global_ref_sigma)
            else:
                print('Please choose a reference potential of the following:\n \
                        {%s,%s,%s}'%('uniform','exp','gaussian'))


                # calculate beta[j] for every observable r_j

        # VERY IMPORTANT: compute reference state self.logZ  for the free energies, so they are properly normalized #
        Z = 0.0
        for rest_index in range(len(ensemble[0])):
            for s in ensemble[rest_index]:
                Z +=  np.exp(-s.free_energy)
        self.logZ = np.log(Z)

        # store this constant so we're not recalculating it all the time in neglogP
        self.ln2pi = np.log(2.0*np.pi)




    def build_exp_ref(self, rest_index,verbose=False):
        """Look at all the structures to find the average observables r_j

        >>    beta_j = np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

        then store this reference potential info for all Restraints of this type for each structure"""


        print( 'Computing parameters for exponential reference potentials...')

        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].nObs  # the number of (model,exp) data values in this restraint
        print('n_observables = ',n_observables)

        distributions = [[] for j in range(n_observables)]
        for s in self.ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
            for j in range(len(s[rest_index].restraints)):
                if verbose == True:
                    print( s[rest_index].restraints[j].model)
                distributions[j].append( s[rest_index].restraints[j].model )
        if verbose == True:
            print('distributions',distributions)

        # Find the MLE average (i.e. beta_j) for each noe
        betas = np.zeros(n_observables)
        for j in range(n_observables):
            # the maximum likelihood exponential distribution fitting the data
            betas[j] =  np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

        # store the beta information in each structure and compute/store the -log P_potential
        for s in self.ensemble:
            s[rest_index].betas = betas
            s[rest_index].compute_neglog_exp_ref()


    def build_gaussian_ref(self, rest_index, use_global_ref_sigma=True,verbose=False):
        """Look at all the structures to find the mean (mu) and std (sigma) of  observables r_j
        then store this reference potential info for all Restraints of this type for each structure"""

    #NOTE: Is this correct?!
        print( 'Computing parameters for Gaussian reference potentials...')

        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].nObs  # the number of (model,exp) data values in this restraint
        print('n_observables = ',n_observables)

        distributions = [[] for j in range(n_observables)]
        for s in self.ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
            for j in range(len(s[rest_index].restraints)):
                if verbose == True:
                    print( s[rest_index].restraints[j].model)
                distributions[j].append( s[rest_index].restraints[j].model )
        if verbose == True:
            print('distributions',distributions)

        # Find the MLE mean (ref_mu_j) and std (ref_sigma_j) for each observable
        ref_mean  = np.zeros(n_observables)
        ref_sigma = np.zeros(n_observables)
        for j in range(n_observables):
            ref_mean[j] =  np.array(distributions[j]).mean()
            squared_diffs = [ (d - ref_mean[j])**2.0 for d in distributions[j] ]
            ref_sigma[j] = np.sqrt( np.array(squared_diffs).sum() / (len(distributions[j])+1.0))
            print(ref_sigma[j])

        if use_global_ref_sigma == True:
            # Use the variance across all ref_sigma[j] values to calculate a single value of ref_sigma for all observables
            global_ref_sigma = ( np.array([ref_sigma[j]**(-2.0) for j in range(n_observables)]).mean() )**-0.5
            for j in range(n_observables):
                ref_sigma[j] = global_ref_sigma
                print(ref_sigma[j])

        # store the ref_mean and ref_sigma information in each structure and compute/store the -log P_potential
        for s in self.ensemble:
            s[rest_index].ref_mean = ref_mean
            s[rest_index].ref_sigma = ref_sigma
            s[rest_index].compute_neglog_gaussian_ref()


    def get_dimensions(self):
        pass



    def construct_matrix(self,verbose=False):
        """ Constructs a matrix with dimensions of nConformations (x) by nSigmas
        and by nGamma parameters from the restraint objects for the output of a
        matrix with numerical values to be sampled """

        Obs = []
        nX = 0
        nGamma = 0
        Matrix = [[],[],[],[],[],[],[],[],[],[]]
        for rest_index in range(len(self.ensemble[0])):
            for s in self.ensemble:
                Obs.append(s[rest_index].n)

                # Start Creating the Matrix:
                Matrix[0].append(nX)
                Matrix[1].append(s[rest_index].Ndof)
                Matrix[2].append(s[rest_index].sigma)
                Matrix[3].append(s[rest_index].allowed_sigma)
                Matrix[4].append(s[rest_index].sigma_index)
                Matrix[5].append(s[rest_index].ref_sigma)
                Matrix[6].append(s[rest_index].betas)

                # Append depending on if there are gamma parameters
                if hasattr(s, 'gamma'):
                    Matrix[7].append(s[rest_index].gamma)
                    Matrix[8].append(s[rest_index].allowed_gamma)
                    Matrix[9].append(s[rest_index].gamma_index)
                    nGamma += 1
                nX += 1

        self.Matrix = np.array(Matrix)
        if verbose == True:
            print('Matrix.shape',self.Matrix.shape)
            print(self.Matrix)


    def neglogP(self, new_state, new_sigma,
            new_gamma_index, verbose=True):
        """Return -ln P of the current configuration."""

        # Current Structure being sampled:
        s = self.ensemble[0][new_state]

        result = s.free_energy + self.logz

        if s.sse != 0:
           result += (s.Ndof)*np.log(new_sigma)  # for use with log-spaced sigma values
           result += s.sse / (2.0*new_sigma**2.0)
           result += (s.Ndof)/2.0*self.ln2pi  # for normalization
           if new_gamme_index != None:
               result += s.sse[new_gamma_index] / (2.0*new_sigma**2.0)

        if verbose:
            print('s = ',s)
            print('Result =',result)

            print('state, f_sim', new_state, s.free_energy,)
            print('s.sse', s.sse, 's.Ndof', s.Ndof)
            print('s.sum_neglog_exp_ref', s.sum_neglog_exp_ref)
            print('s.sum_neglog_gaussian_ref', s.sum_neglog_gaussian_ref)
        return result



    def sample(self, nsteps):
        "Perform nsteps of posterior sampling on the constructed matrix."

        Matrix = self.Matrix
        ## Partition the various degrees of freedom of the Matrix
        # Conformational Space
        X = Matrix[0]
        nX = len(X)
        print('Number of Conformations: ',nX)

        # Degrees of freedom
        Ndof = Matrix[1]

        # Sigma Space
        new_sigma = Matrix[2]
        nSigma = len(new_sigma)
        print('nSigma ',(nSigma))

        # allowed sigma
        Allowed_sigma = Matrix[3]

        # Sigma index
        new_sigma_index = Matrix[4][0]
#        print('Sigma_index ',Sigma_index)

        # ref sigma
        Ref_sigma = Matrix[5]

        # Beta space
        Betas = Matrix[6]

        # Gamma portion of Matrix
        if hasattr(self.ensemble, 'gamma'):
            new_gamma = Matrix[7]
            Allowed_gamma = Matrix[8]
            new_gamma_index = Matrix[9]
        else:
            new_gamma_index = None
            #print('Gamma',new_gamma)

        # Set the state
        new_state = X[0]

        for step in range(nsteps):

            if np.random.random() < 0.16:
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(Allowed_sigma))
                new_sigma = Allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.32:
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(Allowed_sigma))
                new_sigma = Allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.48:
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(Allowed_sigma))
                new_sigma = Allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.60:
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(Allowed_sigma))
                new_sigma = Allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.78:
                if hasattr(self.ensemble, 'gamma'):
                    new_gamma_index +=  (np.random.randint(3)-1)
                    new_gamma_index = new_gamma_index%(len(Allowed_gamma))
                    new_gamma = Allowed_gamma[new_gamma_index]

            elif np.random.random() < 0.99:
                # take a random step in state space
                new_state = np.random.randint(nX)

            else:
                new_state = np.random.randint(nX)

            # compute new "energy"
            verbose = True
            new_E = self.neglogP(new_state, new_sigma, new_gamma_index, verbose=verbose)


            # accept or reject the MC move according to Metroplis criterion
            accept = False
            if new_E < self.E:
                accept = True
            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

#
#      	    # Store trajectory counts
#            self.traj.sampled_sigma_noe[self.sigma_noe_index] += 1
#            self.traj.sampled_sigma_J[self.sigma_J_index] += 1
#	        self.traj.sampled_sigma_cs_H[self.sigma_cs_H_index] += 1
#            self.traj.sampled_sigma_cs_Ha[self.sigma_cs_Ha_index] += 1
#            self.traj.sampled_sigma_cs_N[self.sigma_cs_N_index] += 1
#            self.traj.sampled_sigma_cs_Ca[self.sigma_cs_Ca_index] += 1
#	        self.traj.sampled_sigma_pf[self.sigma_pf_index] += 1
#            self.traj.sampled_gamma[self.gamma_index] += 1
#            self.traj.state_counts[self.state] += 1
#
#            # update parameters
#            if accept:
#                self.E = new_E
#                self.sigma_noe = new_sigma_noe
#                self.sigma_noe_index = new_sigma_noe_index
#                self.sigma_J = new_sigma_J
#                self.sigma_J_index = new_sigma_J_index
#                self.sigma_cs_H = new_sigma_cs_H
#                self.sigma_cs_H_index = new_sigma_cs_H_index
#                self.sigma_cs_Ha = new_sigma_cs_Ha
#                self.sigma_cs_Ha_index = new_sigma_cs_Ha_index
#                self.sigma_cs_N = new_sigma_cs_N
#                self.sigma_cs_N_index = new_sigma_cs_N_index
#                self.sigma_cs_Ca = new_sigma_cs_Ca
#                self.sigma_cs_Ca_index = new_sigma_cs_Ca_index
#                self.sigma_pf = new_sigma_pf
#                self.sigma_pf_index = new_sigma_pf_index
#		        self.gamma = new_gamma
#                self.gamma_index = new_gamma_index
#                self.state = new_state
#                self.ensemble_index = new_ensemble_index
#                self.accepted += 1.0
#                self.total += 1.0
#
#            # store trajectory samples
#            if step%self.traj_every == 0:
#                self.traj.trajectory.append( [int(step), float(self.E), int(accept), int(self.state), int(self.sigma_noe_index), int(self.sigma_J_index), int(self.sigma_cs_H_index), int(self.sigma_cs_Ha_index), int(self.sigma_cs_N_index), int(self.sigma_cs_Ca_index), int(self.sigma_pf_index), int(self.gamma_index)] )
#
#class PosteriorSamplingTrajectory(object):
#    "A class to store and perform operations on the trajectories of sampling runs."
#
#    def __init__(self, ensemble, allowed_sigma_noe, allowed_sigma_J,
#            allowed_sigma_cs_H, allowed_sigma_cs_Ha, allowed_sigma_cs_N,
#            allowed_sigma_cs_Ca, allowed_sigma_pf, allowed_gamma):
#        "Initialize the PosteriorSamplingTrajectory."
#
#        self.nstates = len(ensemble)
#        self.ensemble = ensemble
#
#        print( 'self.ensemble[0] = ',self.ensemble[0])
#        self.nnoe = len(self.ensemble[0].noe_restraints)
#        self.ndihedrals = len(self.ensemble[0].dihedral_restraints)
#	    self.ncs_H = len(self.ensemble[0].cs_H_restraints)
#        self.ncs_Ha = len(self.ensemble[0].cs_Ha_restraints)
#        self.ncs_Ca = len(self.ensemble[0].cs_Ca_restraints)
#        self.ncs_N = len(self.ensemble[0].cs_N_restraints)
#	    self.npf = len(self.ensemble[0].pf_restraints)
#
#        self.allowed_sigma_noe = allowed_sigma_noe
#        self.sampled_sigma_noe = np.zeros(len(allowed_sigma_noe))
#
#        self.allowed_sigma_J = allowed_sigma_J
#        self.sampled_sigma_J = np.zeros(len(allowed_sigma_J))
#
#        self.allowed_sigma_cs_H = allowed_sigma_cs_H
#        self.sampled_sigma_cs_H = np.zeros(len(allowed_sigma_cs_H))
#
#        self.allowed_sigma_cs_Ha = allowed_sigma_cs_Ha
#        self.sampled_sigma_cs_Ha = np.zeros(len(allowed_sigma_cs_Ha))
#
#        self.allowed_sigma_cs_N = allowed_sigma_cs_N
#        self.sampled_sigma_cs_N = np.zeros(len(allowed_sigma_cs_N))
#
#        self.allowed_sigma_cs_Ca = allowed_sigma_cs_Ca
#        self.sampled_sigma_cs_Ca = np.zeros(len(allowed_sigma_cs_Ca))
#
#        self.allowed_sigma_pf = allowed_sigma_pf
#        self.sampled_sigma_pf = np.zeros(len(allowed_sigma_pf))
#
#        self.allowed_gamma = allowed_gamma
#        self.sampled_gamma = np.zeros(len(allowed_gamma))
#
#
#        self.state_counts = np.ones(self.nstates)  # add a pseudocount to avoid log(0) errors
#
#        self.f_sim = np.array([e.free_energy for e in ensemble])
#        self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()
#
#        # stores samples [step, self.E, accept, state, sigma_noe, sigma_J, sigma_cs, gamma]
#        self.trajectory_headers = ['step', 'E', 'accept', 'state', 'sigma_noe_index', 'sigma_J_index', 'sigma_cs_H_index', 'sigma_cs_Ha_index', 'sigma_cs_N_index', 'sigma_cs_Ca_index', 'sigma_pf_index', 'gamma_index']
#        self.trajectory = []
#
#        # a dictionary to store results for YAML file
#        self.results = {}

#
#    def process(self):
#        """Process the trajectory, computing sampling statistics,
#        ensemble-average NMR observables.
#
#        NOTE: Where possible, we convert to lists, because the YAML output
#        is more readable"""
#
#        # Store the trajectory in rsults
#        self.results['trajectory_headers'] = self.trajectory_headers
#        self.results['trajectory'] = self.trajectory
#
#        # Store the nuisance parameter distributions
#        self.results['allowed_sigma_noe'] = self.allowed_sigma_noe.tolist()
#        self.results['allowed_sigma_J'] = self.allowed_sigma_J.tolist()
#        self.results['allowed_sigma_cs_H'] = self.allowed_sigma_cs_H.tolist()
#        self.results['allowed_sigma_cs_Ha'] = self.allowed_sigma_cs_Ha.tolist()
#        self.results['allowed_sigma_cs_N'] = self.allowed_sigma_cs_N.tolist()
#        self.results['allowed_sigma_cs_Ca'] = self.allowed_sigma_cs_Ca.tolist()
#     	self.results['allowed_sigma_pf'] = self.allowed_sigma_pf.tolist()
#        self.results['allowed_gamma'] = self.allowed_gamma.tolist()
#        self.results['sampled_sigma_noe'] = self.sampled_sigma_noe.tolist()
#        self.results['sampled_sigma_J'] = self.sampled_sigma_J.tolist()
#        self.results['sampled_sigma_cs_H'] = self.sampled_sigma_cs_H.tolist()
#        self.results['sampled_sigma_cs_Ha'] = self.sampled_sigma_cs_Ha.tolist()
#        self.results['sampled_sigma_cs_N'] = self.sampled_sigma_cs_N.tolist()
#        self.results['sampled_sigma_cs_Ca'] = self.sampled_sigma_cs_Ca.tolist()
#	    self.results['sampled_sigma_pf'] = self.sampled_sigma_pf.tolist()
#        self.results['sampled_gamma'] = self.sampled_gamma.tolist()
#
#        # Calculate the modes of the nuisance parameter marginal distributions
#        self.results['sigma_noe_mode'] = float(self.allowed_sigma_noe[ np.argmax(self.sampled_sigma_noe) ])
#        self.results['sigma_J_mode']   = float(self.allowed_sigma_J[ np.argmax(self.sampled_sigma_J) ])
#        self.results['sigma_cs_H_mode']   = float(self.allowed_sigma_cs_H[ np.argmax(self.sampled_sigma_cs_H) ])
#        self.results['sigma_cs_Ha_mode']   = float(self.allowed_sigma_cs_Ha[ np.argmax(self.sampled_sigma_cs_Ha) ])
#        self.results['sigma_cs_N_mode']   = float(self.allowed_sigma_cs_N[ np.argmax(self.sampled_sigma_cs_N) ])
#        self.results['sigma_cs_Ca_mode']   = float(self.allowed_sigma_cs_Ca[ np.argmax(self.sampled_sigma_cs_Ca) ])
#     	self.results['sigma_pf_mode']	= float(self.allowed_sigma_pf[ np.argmax(self.sampled_sigma_pf) ])
#        self.results['gamma_mode']     = float(self.allowed_gamma[ np.argmax(self.sampled_gamma) ])
#
#        # copy over the purely computational free energies f_i
#        self.results['comp_f'] = self.f_sim.tolist()
#
#        # Estimate the populations of each state
#        self.results['state_pops'] = (self.state_counts/self.state_counts.sum()).tolist()
#
#        # Estimate uncertainty in the populations by bootstrap
#        self.nbootstraps = 1000
#        self.bootstrapped_state_pops = np.random.multinomial(self.state_counts.sum(), self.results['state_pops'], size=self.nbootstraps)
#        self.results['state_pops_std'] = self.bootstrapped_state_pops.std(axis=0).tolist()
#
#        # Estimate the free energies of each state
#        self.results['state_f'] = (-np.log(self.results['state_pops'])).tolist()
#        state_f = -np.log(self.results['state_pops'])
#        ref_f = state_f.min()
#        state_f -=  ref_f
#        self.results['state_f'] = state_f.tolist()
#        self.bootstrapped_state_f = -np.log(self.bootstrapped_state_pops+1e-10) - ref_f  # add pseudocount to avoid log(0)s in the bootstrap
#        self.results['state_f_std'] = self.bootstrapped_state_f.std(axis=0).tolist()
#
#        # Estimate the ensemble-<r**-6>averaged noe
#        mean_noe = np.zeros(self.nnoe)
#        Z = np.zeros(self.nnoe)
#        for i in range(self.nstates):
#            for j in range(self.nnoe):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].noe_restraints[j].weight
#                r = self.ensemble[i].noe_restraints[j].model_noe
#                mean_noe[j] += pop*weight*(r**(-6.0))
#                Z[j] += pop*weight
#        mean_noe = (mean_noe/Z)**(-1.0/6.0)
#        self.results['mean_noe'] = mean_noe.tolist()
#
#        # compute the experimental noe, using the most likely gamma'
#        exp_noe = np.array([self.results['gamma_mode']*self.ensemble[0].noe_restraints[j].exp_noe \
#                                      for j in range(self.nnoe)])
#        self.results['exp_noe'] = exp_noe.tolist()
#
#        self.results['noe_pairs'] = []
#        for j in range(self.nnoe):
#            pair = [int(self.ensemble[0].noe_restraints[j].i), int(self.ensemble[0].noe_restraints[j].j)]
#            self.results['noe_pairs'].append(pair)
#        abs_diffs = np.abs( exp_noe - mean_noe )
#        self.results['disagreement_noe_mean'] = float(abs_diffs.mean())
#        self.results['disagreement_noe_std'] = float(abs_diffs.std())
#
#        # Estimate the ensemble-averaged J-coupling values
#        mean_Jcoupling = np.zeros(self.ndihedrals)
#        Z = np.zeros(self.ndihedrals)
#        for i in range(self.nstates):
#            for j in range(self.ndihedrals):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].dihedral_restraints[j].weight
#                r = self.ensemble[i].dihedral_restraints[j].model_Jcoupling
#                mean_Jcoupling[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_Jcoupling = (mean_Jcoupling/Z)
#        self.results['mean_Jcoupling'] = mean_Jcoupling.tolist()
#
#        # Compute the experiment Jcouplings
#        exp_Jcoupling = np.array([self.ensemble[0].dihedral_restraints[j].exp_Jcoupling for j in range(self.ndihedrals)])
#        self.results['exp_Jcoupling'] = exp_Jcoupling.tolist()
#        abs_Jdiffs = np.abs( exp_Jcoupling - mean_Jcoupling )
#        self.results['disagreement_Jcoupling_mean'] = float(abs_Jdiffs.mean())
#        self.results['disagreement_Jcoupling_std'] = float(abs_Jdiffs.std())
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_H = np.zeros(self.ncs_H)
#        Z = np.zeros(self.ncs_H)
#        for i in range(self.nstates):
#            for j in range(self.ncs_H):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_H_restraints[j].weight
#                r = self.ensemble[i].cs_H_restraints[j].model_cs_H
#                mean_cs_H[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_H = (mean_cs_H/Z)
#        self.results['mean_cs_H'] = mean_cs_H.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_H = np.array([self.ensemble[0].cs_H_restraints[j].exp_cs_H for j in range(self.ncs_H)])
#        self.results['exp_cs_H'] = exp_cs_H.tolist()
#        abs_cs_H_diffs = np.abs( exp_cs_H - mean_cs_H )
#        self.results['disagreement_cs_H_mean'] = float(abs_cs_H_diffs.mean())
#        self.results['disagreement_cs_H_std'] = float(abs_cs_H_diffs.std())
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_Ha = np.zeros(self.ncs_Ha)
#        Z = np.zeros(self.ncs_Ha)
#        for i in range(self.nstates):
#            for j in range(self.ncs_Ha):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_Ha_restraints[j].weight
#                r = self.ensemble[i].cs_Ha_restraints[j].model_cs_Ha
#                mean_cs_Ha[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_Ha = (mean_cs_Ha/Z)
#        self.results['mean_cs_Ha'] = mean_cs_Ha.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_Ha = np.array([self.ensemble[0].cs_Ha_restraints[j].exp_cs_Ha for j in range(self.ncs_Ha)])
#        self.results['exp_cs_Ha'] = exp_cs_Ha.tolist()
#        abs_cs_Ha_diffs = np.abs( exp_cs_Ha - mean_cs_Ha )
#        self.results['disagreement_cs_Ha_mean'] = float(abs_cs_Ha_diffs.mean())
#        self.results['disagreement_cs_Ha_std'] = float(abs_cs_Ha_diffs.std())
#
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_N = np.zeros(self.ncs_N)
#        Z = np.zeros(self.ncs_N)
#        for i in range(self.nstates):
#            for j in range(self.ncs_N):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_N_restraints[j].weight
#                r = self.ensemble[i].cs_N_restraints[j].model_cs_N
#                mean_cs_N[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_N = (mean_cs_N/Z)
#        self.results['mean_cs_N'] = mean_cs_N.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_N = np.array([self.ensemble[0].cs_N_restraints[j].exp_cs_N for j in range(self.ncs_N)])
#        self.results['exp_cs_N'] = exp_cs_N.tolist()
#        abs_cs_N_diffs = np.abs( exp_cs_N - mean_cs_N )
#        self.results['disagreement_cs_N_mean'] = float(abs_cs_N_diffs.mean())
#        self.results['disagreement_cs_N_std'] = float(abs_cs_N_diffs.std())
#
#
#        # Estimate the ensemble-averaged chemical shift values
#        mean_cs_Ca = np.zeros(self.ncs_Ca)
#        Z = np.zeros(self.ncs_Ca)
#        for i in range(self.nstates):
#            for j in range(self.ncs_Ca):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].cs_Ca_restraints[j].weight
#                r = self.ensemble[i].cs_Ca_restraints[j].model_cs_Ca
#                mean_cs_Ca[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_cs_Ca = (mean_cs_Ca/Z)
#        self.results['mean_cs_Ca'] = mean_cs_Ca.tolist()
#
#        # Compute the experiment chemical shift
#        exp_cs_Ca = np.array([self.ensemble[0].cs_Ca_restraints[j].exp_cs_Ca for j in range(self.ncs_Ca)])
#        self.results['exp_cs_Ca'] = exp_cs_Ca.tolist()
#        abs_cs_Ca_diffs = np.abs( exp_cs_Ca - mean_cs_Ca )
#        self.results['disagreement_cs_Ca_mean'] = float(abs_cs_Ca_diffs.mean())
#        self.results['disagreement_cs_Ca_std'] = float(abs_cs_Ca_diffs.std())
#
#
#
#        # Estimate the ensemble-averaged protection factor values
#        mean_pf = np.zeros(self.npf)
#        Z = np.zeros(self.npf)
#        for i in range(self.nstates):
#            for j in range(self.npf):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[i].pf_restraints[j].weight
#                r = self.ensemble[i].pf_restraints[j].model_pf
#                mean_pf[j] += pop*weight*r
#                Z[j] += pop*weight
#        mean_pf = (mean_pf/Z)
#        self.results['mean_pf'] = mean_pf.tolist()
#
#        # Compute the experiment protection factor
#
#    	exp_pf = np.array([self.ensemble[0].pf_restraints[j].exp_pf for j in range(self.npf)])
##        exp_pf = np.array([self.ensemble[0].pf_restraints[j].exp_pf for j in range(self.npf)])
#        self.results['exp_pf'] = exp_pf.tolist()
#        abs_pfdiffs = np.abs( exp_pf - mean_pf )
#        self.results['disagreement_pf_mean'] = float(abs_pfdiffs.mean())
#        self.results['disagreement_pf_std'] = float(abs_pfdiffs.std())
#
#
    def logspaced_array(self, xmin, xmax, nsteps):
        ymin, ymax = np.log(xmin), np.log(xmax)
        dy = (ymax-ymin)/nsteps
        return np.exp(np.arange(ymin, ymax, dy))


    #NOTE: This will work well with Cython if we go that route.
    # Standardized: Yes ; Binary: Yes; Human Readable: No;

    def write_results(self, outfilename='traj.npz'):
        """Writes a compact file of several arrays into binary format."""

        np.savez_compressed(outfilename, self.results)

    def read_results(self,filename):
        """Reads a npz file"""

        loaded = np.load(filename)
        print( loaded.items())







