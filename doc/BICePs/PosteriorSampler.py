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
from KarplusRelation import *     # Returns J-coupling values from dihedral angles
from Restraint import *
from toolbox import *

##############################################################################
# Main
##############################################################################

class PosteriorSampler(object):
    """A class to perform posterior sampling of conformational populations

    Parameters
    ----------
    ensemble: list
        a list of lists of Restraint objects, one list for each conformation.
    freq_write_traj: int
        the frequency (in steps) to write the MCMC trajectory
    freq_print: int
        the frequency (in steps) to print status
    freq_save_traj: int
        the frequency (in steps) to store the MCMC trajectory"""

    def __init__(self, ensemble, freq_write_traj=1000,
            freq_print=1000, freq_save_traj=100):
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

        # The initial state of the structural ensemble we're sampling from
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
        self.compute_logZ()


    def compute_logZ(self):
        """Compute reference state logZ for the free energies to normalize."""

        Z = 0.0
#        for rest_index in range(len(self.ensemble[0])):
#            for s in self.ensemble[rest_index]:
        for s in self.ensemble:
            Z +=  np.exp(-s[0].free_energy)
        self.logZ = np.log(Z)
        self.ln2pi = np.log(2.0*np.pi)


    def build_exp_ref(self, rest_index, verbose=False):
        """Look at all the structures to find the average observables r_j

        >>  beta_j = np.array(distributions[j]).sum()/(len(distributions[j])+1.0)

        then store this reference potential info for all Restraints of this
        type for each structure"""

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
        print('distributions ,',distributions,np.array(distributions).shape)

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


    def build_gaussian_ref(self, rest_index, use_global_ref_sigma=False, verbose=False):
        """Look at all the structures to find the mean (mu) and std (sigma)
        of  observables r_j then store this reference potential info for all
        Restraints of this type for each structure"""

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

        print('ref_mean',ref_mean)
        print('ref_sigma',ref_sigma)

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


    def compile_nuisance_parameters(self, verbose=False):
        """Compiles arrays into a list for each nuisance parameter.

        Returns
        -------

        [[allowed_sigma_cs_H],[allowed_sigma_noe,allowed_gamma_noe],...,[Nth_restraint]]"""

        # Generate empty lists for each restraint to fill with nuisance parameters
        nuisance_para = [ ]

        for rest_index in range(len(self.ensemble[0])):
            s_r = self.ensemble[0][rest_index]

            # Get nuisance parameters for this specific restraint
            nuisance_para.append([
                    getattr(s_r, para) for para in s_r._nuisance_parameters])

        self.nuisance_para = nuisance_para
        # Construct the matrix converting all values to floats for C++
        if verbose == True:
            print(self.nuisance_para)
            print('Number of Restraints = ',len(self.nuisance_para))
            print('Number of nuisance parameters for each:\n ')
            for i in range(len(nuisance_para)):
                print(len(self.nuisance_para[i]))
        np.save('compiled_nuisance_parameters.npy',self.nuisance_para)


    def neglogP(self, new_state, parameters, parameter_indices, verbose=True):
        """Return -ln P of the current configuration.

        Parameters
        ----------
        new_state: int
            the new conformational state from Sample()
        parameters: list
            a list of the new parameters for each of the restraints
        parameter_indices: list
            a list of the new indices for each of the parameters

        Returns
        -------
        Energy
            the energy
        """

        # Current Structure being sampled (list of restraint objects):
        s = self.ensemble[int(new_state)]

        # Grab the free energy of the state and normalize
        result = s[0].free_energy + self.logZ

        # Use the restraint index to get the corresponding sigma.
        for rest_index in range(len(s)):

            # Use with log-spaced sigma values
            result += (s[rest_index].Ndof)*np.log(parameters[rest_index][0])

            # Is gamma a parameter we need to consider?
            if 'allowed_gamma' in s[rest_index]._nuisance_parameters:
                result += s[rest_index].sse[int(parameter_indices[rest_index][1])] / (2.0*parameters[rest_index][0]**2.0)

            else:
                result += s[rest_index].sse / (2.0*float(parameters[rest_index][0])**2.0)

            result += (s[rest_index].Ndof)/2.0*self.ln2pi  # for normalization

            # Which reference potential was used for each restraint?
            if hasattr(s[rest_index], 'sum_neglog_exp_ref'):
                result -= s[rest_index].sum_neglog_exp_ref

            if hasattr(s[rest_index], 'sum_neglog_gaussian_ref'):
                result -= s[rest_index].sum_neglog_gaussian_ref

            if verbose:
                print('\nstep = ',int(self.total+1))
                print('s[%s] = '%(rest_index),s[rest_index])
                print('Result =',result)
                print('state %s, f_sim %s'%(new_state, s[rest_index].free_energy))
                if 'allowed_gamma' in s[rest_index]._nuisance_parameters:
                     print('s[%s].sse[%s]'%(rest_index,int(parameter_indices[rest_index][1])), s[rest_index].sse[int(parameter_indices[rest_index][1])], 's[%s].Ndof'%rest_index, s[rest_index].Ndof)
                else:
                     print('s[%s].sse'%rest_index, s[rest_index].sse, 's[%s].Ndof'%rest_index, s[rest_index].Ndof)
                if hasattr(s[rest_index], 'sum_neglog_exp_ref'):
                    print('s[%s].sum_neglog_exp_ref'%rest_index, s[rest_index].sum_neglog_exp_ref)
                if hasattr(s[rest_index], 'sum_neglog_gaussian_ref'):
                    print('s[%s].sum_neglog_gaussian_ref'%rest_index, s[rest_index].sum_neglog_gaussian_ref)
        if verbose:
            print('######################################################')
        return result


    def sample(self, nsteps, verbose=True):
        """Perform n number of steps (nsteps) of posterior sampling, where Monte
        Carlo moves are accepted or rejected according to Metroplis criterion."""

        # Generate random restraint index to initialize the sigma and gamma parameters
        self.new_rest_index = np.random.randint(len(self.ensemble[0]))

        # Initialize the state
        self.new_state = int(self.state)

        # Store a list of parameter indices for each restraint inside a list
        # parameter_indices e.g., [[161], [142], ...]
        self.parameter_indices = [ ]

        # Store a list of parameters that correspond to the index inside a list
        # parameters e.g., [[1.2122652], [0.832136160], ...]
        self.parameters = [ ]
        # Loop through the restraints, and get the parameters and indices
        for rest_index in range(len(self.ensemble[self.new_state])):
            Restraint = self.ensemble[self.new_state][rest_index]
            self.parameter_indices.append(
                    [getattr(Restraint, para) for para in Restraint._parameter_indices])
            self.parameters.append(
                    [getattr(Restraint, para) for para in Restraint._parameters])


        # The new parameter index to use in initial step (this is for restraints with more than one nuisance parameters)
        self.new_para_index = np.random.randint(
                len(self.parameter_indices[self.new_rest_index]) )

        # RAND = generalized probability of taking a step in restraint space
        #.. given the total number of restraints.
        RAND = 1. - 1./(len(self.ensemble[0]) + 1.)

        for step in range(nsteps):

            # Redefine based upon acceptance (Metroplis criterion)
            new_state = self.new_state

            # Store the randomly generated new restraint index for each step
            new_rest_index =  self.new_rest_index

            # Store the randomly generated new parameter index for each step
            new_para_index = self.new_para_index

            # parameter_indices e.g. [[161], [142]]
            parameter_indices = self.parameter_indices

            # parameters e.g. [[1.2122652], [0.832136160]]
            parameters = self.parameters

            # Now, select the specific parameter index from the list of
            #.. parameter indices (parameter_indices) given the restraint index
            #.. and the parameter index
            index = parameter_indices[new_rest_index][new_para_index]

            # Get the specific nuisance parameter from the compiled list
            # self.nuisance_para = [ [allowed_sigma_cs_H],
            #..   [allowed_sigma_noe, allowed_gamma_noe], ... [Nth_restraint] ]
            nuisance_para = self.nuisance_para[new_rest_index][new_para_index]

            if np.random.random() < RAND:
                ## Take a random step in the space of specific parameter
                # Shift the index by +1, 0 or -1
                index += (np.random.randint(3)-1)
                # New index for specific parameter that belongs to a specific restraint
                index = index%(len(nuisance_para))

                ## Temporary replacement until satisfied by Metroplis criterion
                # Replace the old parameter with the new parameter
                parameters[new_rest_index][new_para_index] = nuisance_para[index]
                # Replace the old index with the new index
                parameter_indices[new_rest_index][new_para_index] = index

            else:
                ## Take a random step in state space
                new_state = np.random.randint(self.nstates)

            if verbose:
                print('*****************************************')
                print('new_rest_index ', new_rest_index )
                print('new_para_index ', new_para_index )
                print('new_state ', new_state )
                print('new_sigma ', parameters[new_rest_index][0] )
                print('new_sigma_index ', parameter_indices[new_rest_index][0] )
                print('index ', index)
                print('new_allowed_sigma ', nuisance_para )
                print('parameter_indices ', parameter_indices )
                print('parameters ',parameters)
                print('*****************************************')

            # Compute new "energy"
            new_E = self.neglogP(new_state, parameters, parameter_indices, verbose=True)

            # Accept or reject the MC move according to Metroplis criterion
            accept = False

            if new_E < self.E:
                accept = True

            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

      	    # Store trajectory state counts
            self.traj.state_counts[new_state] += 1

            # Store the counts of sampled sigma along the trajectory
            for i in range(len(self.ensemble[self.new_state])):
                self.traj.sampled_sigmas[i][parameter_indices[i][0]] += 1
            #self.traj.sampled_sigmas[new_rest_index][parameter_indices[new_rest_index][0]] += 1

            # If we are sampling gamma, then store along the trajectory
            for i in range(len(self.ensemble[self.new_state])):
                if hasattr(self.ensemble[new_state][i], 'gamma'):
                    self.traj.sampled_gamma[parameter_indices[i][1]] += 1
                    #self.traj.sampled_gamma[parameter_indices[new_rest_index][1]] += 1

            # Update parameters based upon acceptance (Metroplis criterion)
            if accept:
                self.E = new_E
                self.new_state = new_state
                self.parameter_indices = parameter_indices
                self.parameters = parameters
                self.new_rest_index = new_rest_index
                self.new_para_index = new_para_index
                self.accepted += 1.0
            self.total += 1.0

            if verbose:
                if accept:
                    print('*****************************************')
                    print('self.E', self.E)
                    print('self.new_state ', self.new_state )
                    print('self.new_rest_index ', self.new_rest_index )
                    print('self.new_para_index ', new_para_index)
                    print('self.accepted', self.accepted)
                    print('*****************************************')

            #NOTE: There will need to be additional parameters here for protection factor.


            # Store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step+1), float(self.E),
                    int(accept), int(self.new_state),
                    np.array([self.parameter_indices[i] for i in range(len(self.parameter_indices))])])

            # Randomly generate new restraint index for the next step
            self.new_rest_index = np.random.randint(len(self.ensemble[0]))

            # Randomly generate new para index for the next step
            self.new_para_index = np.random.randint(
                    len(self.parameter_indices[self.new_rest_index]) )

        print('\nAccepted %s %% \n'%(self.accepted/self.total*100.))


class PosteriorSamplingTrajectory(object):
    """A container class to store and perform operations on the trajectories of
    sampling runs."""

    def __init__(self, ensemble):
        """Initialize the PosteriorSamplingTrajectory container class."""

        self.ensemble = ensemble
        self.nstates = len(self.ensemble)
        self.state_counts = np.ones(self.nstates)  # add a pseudo-count to avoid log(0) errors

        # Lists for each restraint inside a list
        self.sampled_sigmas = [ [] for i in range(len(ensemble[0])) ]
        self.allowed_sigmas = [ [] for i in range(len(ensemble[0])) ]
        f_sim = []
        rest_index = 0
        for s in ensemble:
            f_sim.append(s[0].free_energy)
            for rest_index in range(len(s)):
                allowed_sigma = s[rest_index].allowed_sigma
                if rest_index < len(self.sampled_sigmas):
                    self.sampled_sigmas[rest_index] = np.zeros(len(allowed_sigma))
                    self.allowed_sigmas[rest_index] = allowed_sigma
                    # If the restraint has a gamma parameter, then construct a container
                    if hasattr(s[rest_index], 'gamma'):
                        self.allowed_gamma = s[rest_index].allowed_gamma
                        self.sampled_gamma = list(np.zeros(len(self.allowed_gamma)))
                rest_index += 1

        self.f_sim = np.array(f_sim)
        self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()

        # Generate a list of the names of the parameter indices for the traj header
        parameter_indices = []
        s = self.ensemble[0]
        for rest_index in range(len(s)):
            parameter_indices.append( getattr(s[rest_index], '_parameter_indices') )

        self.trajectory_headers = ["step", "E", "accept", "state",
                "para_index = %s"%parameter_indices]

        self.trajectory = []

        self.results = {}

    def process(self):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables."""

        # Store the trajectory in results
        self.results['trajectory_headers'] = self.trajectory_headers
        self.results['trajectory'] = self.trajectory

        # Store the nuisance parameter distributions
        self.results['allowed_sigma'] = self.allowed_sigmas
        self.results['sampled_sigma'] = self.sampled_sigmas

        self.results['allowed_gamma'] = None

        for s in self.ensemble:
            for rest_index in range(len(s)):
                if hasattr(s[rest_index], 'gamma'):
                    self.results['sampled_gamma'] = self.sampled_gamma
                    self.results['allowed_gamma'] = self.allowed_gamma

        # Calculate the modes of the nuisance parameter marginal distributions
        self.results['sigma_mode'] = [ float(self.allowed_sigmas[i][ np.argmax(
            self.sampled_sigmas[i]) ]) for i in range(len(self.sampled_sigmas)) ]

        if self.results['allowed_gamma'] is not None:
            self.results['gamma_mode'] = float(self.allowed_gamma[ np.argmax(self.sampled_gamma) ])

        # copy over the purely computational free energies f_i
        self.results['comp_f'] = self.f_sim.tolist()

        # Estimate the populations of each state
        self.results['state_pops'] = (self.state_counts/self.state_counts.sum()).tolist()

        # Estimate uncertainty in the populations by bootstrap
        self.nbootstraps = 1000
        self.bootstrapped_state_pops = np.random.multinomial(self.state_counts.sum(),
                self.results['state_pops'], size=self.nbootstraps)
        self.results['state_pops_std'] = self.bootstrapped_state_pops.std(axis=0).tolist()

        # Estimate the free energies of each state
        self.results['state_f'] = (-np.log(self.results['state_pops'])).tolist()
        state_f = -np.log(self.results['state_pops'])
        ref_f = state_f.min()
        state_f -=  ref_f
        self.results['state_f'] = state_f.tolist()
        self.bootstrapped_state_f = -np.log(self.bootstrapped_state_pops+1e-10) - ref_f  # add pseudo-count to avoid log(0)s in the bootstrap
        self.results['state_f_std'] = self.bootstrapped_state_f.std(axis=0).tolist()

        # Estimate the ensemble-averaged restraint values
#        mean = [ np.zeros(len(self.ensemble[0][rest_index].restraints))
#                for rest_index in range(len(self.ensemble[0])) ]

#        Z = [ np.zeros(len(self.ensemble[0][rest_index].restraints))
#                for rest_index in range(len(self.ensemble[0])) ]

#        for rest_index in range(len(self.ensemble[0])):
#            for i in range(len(self.ensemble[0][rest_index].restraints)):
#                pop = self.results['state_pops'][i]
#                weight = self.ensemble[0][rest_index].restraints[i].weight
#                model = self.ensemble[0][rest_index].restraints[i].model
#                mean[rest_index][i] += pop*weight*model
#                Z[rest_index][i] += pop*weight
#        MEAN = []
#        for i in range(len(mean)):
#            mean = (mean[i]/Z[i])**(-1.0/6.0)
#            MEAN.append(mean)
#        self.results['mean'] = MEAN


    def logspaced_array(self, xmin, xmax, nsteps):
        ymin, ymax = np.log(xmin), np.log(xmax)
        dy = (ymax-ymin)/nsteps
        return np.exp(np.arange(ymin, ymax, dy))

    def write_results(self, outfilename='traj.npz'):
        """Writes a compact file of several arrays into binary format.
        Standardized: Yes ; Binary: Yes; Human Readable: No;"""

        np.savez_compressed(outfilename, self.results)

    def read_results(self,filename):
        """Reads a npz file"""

        loaded = np.load(filename)
        print( loaded.items())







__all__ = [
    'PosteriorSampler',
    'PosteriorSampler.neglogP',
    'PosteriorSampler.logspaced_array',
    'PosteriorSampler.write_results',
    'PosteriorSampler.read_results',
    'PosteriorSamplingTrajectory'

]
