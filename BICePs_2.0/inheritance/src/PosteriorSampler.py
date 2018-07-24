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
        self.traj = PosteriorSamplingTrajectory(self.ensemble)

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

        # Compute ref state logZ for the free energies to normalize.
        self.logZ()


    def logZ(self):
        """Compute reference state logZ for the free energies to normalize."""

        Z = 0.0
        for rest_index in range(len(self.ensemble[0])):
            for s in self.ensemble[rest_index]:
                Z +=  np.exp(-s.free_energy)
        self.logZ = np.log(Z)
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
        # calculate beta[j] for every observable r_j
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


    def construct_matrix(self,verbose=False):
        """ Constructs a matrix with dimensions of nConformations (x) by nSigmas
        and by nGamma parameters from the restraint objects for the output of a
        matrix with numerical values to be sampled """

        Matrix = [ [] for i in range(len(self.ensemble[0])) ]
        for s in self.ensemble:
            for rest_index in range(len(Matrix)):

                # Start Creating the Matrix:
                s_r = s[rest_index]

                # Are there are gamma parameters?
                if hasattr(s, 'gamma'):
                    Matrix[rest_index].append([ [s_r.Ndof],
                        [s_r.sigma,s_r.allowed_sigma,s_r.sigma_index],
                        [s_r.gamma,s_r.allowed_gamma,s_r.gamma_index] ])
                else:
                    Matrix[rest_index].append([ [s_r.Ndof],
                        [s_r.sigma,s_r.allowed_sigma,s_r.sigma_index] ])

        self.Matrix = np.array(Matrix)
        if verbose == True:
            print('Matrix.shape',self.Matrix.shape)
            print(self.Matrix)
        np.save('Matrix.npy',self.Matrix)


    def neglogP(self, new_state, new_rest_index, new_sigma,
            new_gamma_index, verbose=True):
        """Return -ln P of the current configuration."""

        # Current Structure being sampled:
        s = self.ensemble[new_state][new_rest_index]

        result = s.free_energy + self.logZ

        if s.sse != 0:
           result += (s.Ndof)*np.log(new_sigma)  # for use with log-spaced sigma values
           result += s.sse / (2.0*new_sigma**2.0)
           result += (s.Ndof)/2.0*self.ln2pi  # for normalization
           if new_gamma_index != None:
               result += s.sse[new_gamma_index] / (2.0*new_sigma**2.0)

        if verbose:
            print('s = ',s)
            print('Result =',result)
            print('state %s, f_sim %s'%(new_state, s.free_energy))
            print('s.sse', s.sse, 's.Ndof', s.Ndof)
            if hasattr(s, 'sum_neglog_exp_ref'):
                print('s.sum_neglog_exp_ref', s.sum_neglog_exp_ref)
            if hasattr(s, 'sum_neglog_gaussian_ref'):
                print('s.sum_neglog_gaussian_ref', s.sum_neglog_gaussian_ref)
        return result



    def sample(self, nsteps):
        "Perform nsteps of posterior sampling on the constructed matrix."

        Matrix = self.Matrix
        self.write_results()
        #### Partition the Matrix ####
        ## Conformational Space
        new_rest_index = 0
        # Set the state to the first state
        new_state = self.state  # which is set to 0
        s = Matrix[new_rest_index][new_state]

        ## Sigma Space
        sigma_space = s[1]
        new_sigma = sigma_space[0]
        allowed_sigma = sigma_space[1]
        new_sigma_index = sigma_space[2]

        ## Gamma Space
        if hasattr(self.ensemble, 'gamma'):
            gamma_space = s[2]
            new_gamma = gamma_space[0]
            allowed_gamma = gamma_space[1]
            new_gamma_index = gamma_space[2]
        else:
            new_gamma_index = None

        ## Degrees of Freedom
        Ndof = s[0]

        for step in range(nsteps):

            if np.random.random() < 0.16:
                # Sample in sigma space
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(allowed_sigma))
                new_sigma = allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.32:
                # Sample in sigma space
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(allowed_sigma))
                new_sigma = allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.48:
                # Sample in sigma space
                new_sigma_index +=  (np.random.randint(3)-1)
                new_sigma_index = new_sigma_index%(len(allowed_sigma))
                new_sigma = allowed_sigma[new_sigma_index]

            elif np.random.random() < 0.60:
                # Sample in restraint space
                new_rest_index = np.random.randint(len(Matrix))

            elif np.random.random() < 0.78:
                # Sample in gamma space
                if hasattr(self.ensemble, 'gamma'):
                    new_gamma_index +=  (np.random.randint(3)-1)
                    new_gamma_index = new_gamma_index%(len(allowed_gamma))
                    new_gamma = allowed_gamma[new_gamma_index]

            elif np.random.random() < 0.99:
                # take a random step in state space
                new_state = np.random.randint(self.nstates)

            else:
                # take a random step in state space
                new_state = np.random.randint(self.nstates)

            # compute new "energy"
            verbose = True

            new_E = self.neglogP(new_state, new_rest_index, new_sigma, new_gamma_index, verbose=verbose)

            # accept or reject the MC move according to Metroplis criterion
            accept = False

            if new_E < self.E:
                accept = True
            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

      	   # # Store trajectory counts
           # self.traj.sampled_sigma[new_sigma_index] += 1
           # self.traj.state_counts[self.state] += 1
           # if hasattr(self.ensemble, 'gamma'):
           #     self.traj.sampled_gamma[self.gamma_index] += 1

            # update parameters
            if accept:
                self.E = new_E
                self.state = new_state
                self.accepted += 1.0
                self.total += 1.0

            # store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step), float(self.E),
                    int(accept), int(self.state), int(new_sigma_index),
                    new_gamma_index] )


    def write_results(self, outfilename='Matrix.npz'):
       """Writes a compact file of several arrays into binary format."""

       np.savez_compressed(outfilename, self.Matrix)



class PosteriorSamplingTrajectory(object):
    "A class to store and perform operations on the trajectories of sampling runs."

    def __init__(self, ensemble):
        "Initialize the PosteriorSamplingTrajectory."


        #self.f_sim = np.array([e.free_energy for e in ensemble])
        #self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()

        self.trajectory_headers = ['step', 'E', 'accept', 'state',
                'sigma_index', 'gamma_index']

        self.trajectory = []

        # a dictionary to store results for YAML file
        self.results = {}

    def process(self):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables.
        NOTE: Where possible, we convert to lists, because the YAML output
        is more readable"""

        # Store the trajectory in rsults
        self.results['trajectory_headers'] = self.trajectory_headers
        self.results['trajectory'] = self.trajectory

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







