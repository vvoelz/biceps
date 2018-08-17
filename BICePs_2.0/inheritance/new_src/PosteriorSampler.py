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

    INPUTS

        ensemble        - a list of lists of Restraint objects, one list for each conformation.

    OPTIONS

        freq_write_traj - the frequency (in steps) to write the MCMC trajectory
        freq_print      - the frequency (in steps) to print status
        freq_save_traj  - the frequency (in steps) to store the MCMC trajectory

    HISTORY of changes

    """

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


    def compile_nuisance_parameters(self,verbose=False):
        """ Compiles arrays into a list for each nuisance parameter.
        This list will be passed into sample() for each array to have its own
        degrees of freedom. """

        # Generate empty lists for each restraint to fill with nuisance parameters
        nuisance_para = [ [] for i in range(len(self.ensemble[0])) ]

        for s in self.ensemble:
            for rest_index in range(len(self.ensemble[0])):
                # s_r is a specific restraint for a specific state
                s_r = s[rest_index]

                # Get nuisance parameters for this specific restraint
                nuisance_para[rest_index].append([
                        getattr(s_r, para) for para in s_r.nuisance_parameters
                        ])

        self.nuisance_para = nuisance_para
        if verbose == True:
            print(self.nuisance_para)
            print('Number of Restraints = ',len(self.nuisance_para))
            print('Number of nuisance parameters for each:\n ')
            for i in range(len(nuisance_para)):
                print(len(self.nuisance_para[i]))
        np.save('compiled_nuisance_parameters.npy',self.nuisance_para)



    def neglogP(self, new_state, parameters, para_indices, verbose=True):
        """Return -ln P of the current configuration.
        INPUTS
        -------
            new_state    - the new conformational state from Sample()
            parameters   - a list of the new parameters for each of the restraints
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
            if 'allowed_gamma' in s[rest_index].nuisance_parameters:
                result += s[rest_index].sse[int(para_indices[rest_index][1])] / (2.0*parameters[rest_index][0]**2.0)

            else:
                result += s[rest_index].sse / (2.0*parameters[rest_index][0]**2.0)

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
        self.new_state = self.state

        # Store lists for parameters and parameter indices for each restraint
        self.para_indices = [ [] for i in range(len(self.ensemble[0])) ]
        self.parameters = [ [] for i in range(len(self.ensemble[0])) ]

        # Loop through the restraints, store sigmas and tell me the parameters
        for rest_index in range(len(self.ensemble[self.new_state])):
            # Generate the Restraint object
            Restraint = self.ensemble[self.new_state][rest_index]
            # Get parameters for each restraint
            self.parameters[rest_index].append(Restraint.sigma)
            # Get parameter indices for each
            self.para_indices[rest_index].append(Restraint.sigma_index)
            # Get the names of the parameters we plan to sample over
            para = Restraint.nuisance_parameters
            # How many parameters are there for a given restraint?
            if len(para) > 1 :
                if len(para) < 3:
                    self.para_indices[rest_index].append(Restraint.gamma_index)
                    self.parameters[rest_index].append(Restraint.gamma)
                else:
                    print(para)
        # Generate a new parameter index (random)
        self.new_para_index = np.random.randint(
                len(self.para_indices[self.new_rest_index]) )
        RAND = 1. - 1./(len(self.ensemble[0])+1.)
        for step in range(nsteps):

            # Store the randomly generated new restraint index for each step
            new_rest_index =  self.new_rest_index

            # Store the randomly generated new parameter index for each step
            new_para_index = self.new_para_index

            # Redefine based upon acceptance (Metroplis criterion)
            new_state = self.new_state
            para_indices = self.para_indices
            parameters = self.parameters

            # Get the nuisance parameters from the compiled list
            nuisance = self.nuisance_para[new_rest_index][new_state][new_para_index]

            # Get the index of the parameter to a specific restraint
            new_index = para_indices[new_rest_index][new_para_index]

            if np.random.random() < RAND:
                # Shift the index by +1, 0 or -1
                new_index += (np.random.randint(3)-1)
                # New index for the specific restraint and specific parameter
                new_index = new_index%(len(nuisance))
                # Replace the old para with the new para that corresponds to a specific restraint
                parameters[new_rest_index][new_para_index] = nuisance[new_index]
                # Replace the old index with the new index that corresponds to a specific restraint
                para_indices[new_rest_index][new_para_index] = new_index
            else:
                # Take a random step in state space
                new_state = np.random.randint(self.nstates)

            if verbose:
                print('*****************************************')
                print('new_rest_index ', new_rest_index )
                print('new_para_index ', new_para_index )
                print('new_state ', new_state )
                print('new_sigma ', parameters[new_rest_index][0] )   # do not need. Only for testing
                print('new_sigma_index ', para_indices[new_rest_index][0] ) # do not need. Only for testing
                print('new_index ', new_index)
                print('new_allowed_sigma ', nuisance )
                print('para_indices ', para_indices )
                print('parameters ',parameters)
                print('*****************************************')

            # Compute new "energy"
            new_E = self.neglogP(new_state, parameters, para_indices, verbose=True)

            # Accept or reject the MC move according to Metroplis criterion
            accept = False

            if new_E < self.E:
                accept = True

            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True

            # Update parameters based upon acceptance (Metroplis criterion)
            if accept:
                self.E = new_E
                self.new_state = new_state
                self.para_indices[new_rest_index][new_para_index] = new_index
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
                    print('new_sigma ', self.parameters[new_rest_index][0] ) # do not need. Only for testing
                    print('self.accepted', self.accepted)
                    print('*****************************************')

      	    # Store trajectory counts
            self.traj.state_counts[self.new_state] += 1

            if self.new_para_index == 0 :
                self.traj.sampled_sigmas[self.new_rest_index][self.para_indices[self.new_rest_index][self.new_para_index]] += 1
            else: #NOTE: this will need to be changed when there is PF data
                self.traj.sampled_gamma[self.para_indices[self.new_rest_index][self.new_para_index]] += 1

            # Get each parameter index and place in a list
            _parameter_indices = [ ]
            for rest_index in range(len(self.ensemble[0])):
                _parameter_indices.append([ self.para_indices[rest_index][para_index]
                    for para_index in range(len(self.ensemble[0][rest_index].nuisance_parameters)) ])

            # Store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step+1), float(self.E),
                    int(accept), int(self.new_state),
                    list(_parameter_indices)] )

            # Randomly generate new restraint index for the next step
            self.new_rest_index = np.random.randint(len(self.ensemble[0]))

            # Randomly generate new parameter index for the next step
            self.new_para_index = np.random.randint(
                    len(self.para_indices[self.new_rest_index]) )

        print('\nAccepted %s %% \n'%(self.accepted/self.total*100.))


class PosteriorSamplingTrajectory(object):
    """A container class to store and perform operations on the trajectories of
    sampling runs."""

    def __init__(self, ensemble):
        """Initialize the PosteriorSamplingTrajectory container class."""

        self.ensemble = ensemble
        self.nstates = len(self.ensemble)
        self.state_counts = np.ones(self.nstates)  # add a pseudo-count to avoid log(0) errors

        # Create lists of lists to store the sampled sigma parameters for each restraint
        self.sampled_sigmas = [ [] for i in range(len(ensemble[0])) ]
        self.allowed_sigmas = [ [] for i in range(len(ensemble[0])) ]
        f_sim = []
        rest_index = 0
        for s in ensemble:
            for rest_index in range(len(s)):
                f_sim.append(s[rest_index].free_energy)
                allowed_sigma = s[rest_index].allowed_sigma
                if rest_index < len(self.sampled_sigmas):
                    self.sampled_sigmas[rest_index] = np.zeros(len(allowed_sigma))
                    self.allowed_sigmas[rest_index] = allowed_sigma
                    # If there's a gamma parameter, then store the sampled gamma inside a list
                    if hasattr(s[rest_index], 'gamma'):
                        self.allowed_gamma = s[rest_index].allowed_gamma
                        self.sampled_gamma = list(np.zeros(len(self.allowed_gamma)))
                rest_index += 1

        self.f_sim = np.array(f_sim)
        self.sim_pops = np.exp(-self.f_sim)/np.exp(-self.f_sim).sum()

        # Generate a header for the trajectory of sampling
        self.trajectory_headers = ['step', 'E', 'accept', 'state',
                'sigma_index', 'para_index']

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

        if self.results['allowed_gamma'] != None:
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
        mean = [ np.zeros(len(self.ensemble[0][rest_index].restraints))
                for rest_index in range(len(self.ensemble[0])) ]

        Z = [ np.zeros(len(self.ensemble[0][rest_index].restraints))
                for rest_index in range(len(self.ensemble[0])) ]

        for rest_index in range(len(self.ensemble[0])):
            for i in range(len(self.ensemble[0][rest_index].restraints)):
                pop = self.results['state_pops'][i]
                weight = self.ensemble[0][rest_index].restraints[i].weight
                model = self.ensemble[0][rest_index].restraints[i].model
                mean[rest_index][i] += pop*weight*model
                Z[rest_index][i] += pop*weight
        MEAN = []
        for i in range(len(mean)):
            mean = (mean[i]/Z[i])**(-1.0/6.0)
            MEAN.append(mean)
        self.results['mean'] = MEAN


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







