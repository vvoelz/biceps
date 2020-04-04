# -*- coding: utf-8 -*-
import os, sys, glob, copy
import numpy as np
from scipy  import loadtxt, savetxt
from .KarplusRelation import *     # Returns J-coupling values from dihedral angles
from .Restraint import *
from .toolbox import *

class PosteriorSampler(object):
    """
    A class to perform posterior sampling of conformational populations.

    :param list ensemble: a list of lists of Restraint objects, one list for each conformation.
    :param int freq_write_traj: the frequency (in steps) to write the MCMC trajectory
    :param int freq_print: the frequency (in steps) to print status
    :param int freq_save_traj: the frequency (in steps) to store the MCMC trajectory
    """

    def __init__(self, ensemble, freq_write_traj=100., freq_save_traj=100., verbose=False):
        """Initialize PosteriorSampler Class."""

        # Allow the ensemble to pass through the class
        self.ensemble = ensemble
        # Step frequencies to write trajectory info
        self.write_traj = freq_write_traj
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
        # for each Restraint, calculate global reference potential parameters
        # ..by looking across all structures
        #

        # TODO: can't this be more general?!?
        for i,R in enumerate(ensemble[0]):
            if R.ref == "uniform":
                self.traj.ref[i].append('Nan')
                pass
            elif R.ref == 'exp':
                if hasattr(R, 'precomputed'):
                    if not R.precomputed:
                        self.build_exp_ref_pf(i)
                else:
                    self.build_exp_ref(i)
                self.traj.ref[i].append(self.betas)
            elif R.ref == 'gaussian':
                if not R.precomputed:
                    self.build_gaussian_ref_pf(i, use_global_ref_sigma=R.use_global_ref_sigma)
                else:
                    self.build_gaussian_ref(i, use_global_ref_sigma=R.use_global_ref_sigma)
                self.traj.ref[i].append(self.ref_mean)
                self.traj.ref[i].append(self.ref_sigma)
            else:
                raise ValueError('Please choose a reference potential of the following:\n \
                    {%s,%s,%s}'%('uniform','exp','gaussian'))

        # Compute ref state logZ for the free energies to normalize.
        self.compute_logZ()

        self.verbose = verbose

    def compute_logZ(self):
        """Compute reference state logZ for the free energies to normalize."""

        Z = 0.0
        for s in self.ensemble:
            Z +=  np.exp( -np.array(s[0].energy, dtype=np.float128) )
        self.logZ = np.log(Z)
        self.ln2pi = np.log(2.0*np.pi)


    def build_exp_ref(self, rest_index, verbose=False):
        """Look at all the structures to find the average observables r_j, then
        store the reference potential info for all Restraints of this type for
        each structure.

        :math:`beta_j = np.array(distributions[j]).sum()/(len(distributions[j])+1.0)`

        :param int rest_index: index of the restraint"""

        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
        distributions = [[] for j in range(n_observables)]
        for s in self.ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
            for j in range(len(s[rest_index].restraints)):
                if verbose == True:
                    print( s[rest_index].restraints[j]['model'])
                distributions[j].append( s[rest_index].restraints[j]['model'] )
        if verbose == True:
            print('distributions',distributions)
        # Find the MLE average (i.e. beta_j) for each noe
        # calculate beta[j] for every observable r_j
        self.betas = np.zeros(n_observables)
        for j in range(n_observables):
            # the maximum likelihood exponential distribution fitting the data
            self.betas[j] =  np.array(distributions[j]).sum()/(len(distributions[j])+1.0)
        # store the beta information in each structure and compute/store the -log P_potential
        for s in self.ensemble:
            s[rest_index].betas = self.betas
            s[rest_index].compute_neglog_exp_ref()


    def build_gaussian_ref(self, rest_index, use_global_ref_sigma=False, verbose=False):
        """Look at all the structures to find the mean (mu) and std (sigma) of
        observables r_j then store this reference potential info for all
        restraints of this type for each structure.

        :param int rest_index: index of the restraint
        :param bool default=False use_global_ref_sigma:
        """

        #print( 'Computing parameters for Gaussian reference potentials...')
        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
        #print('n_observables = ',n_observables)
        distributions = [[] for j in range(n_observables)]
        for s in self.ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
            for j in range(len(s[rest_index].restraints)):
                if verbose == True:
                    print( s[rest_index].restraints[j]['model'])
                distributions[j].append( s[rest_index].restraints[j]['model'] )
        if verbose == True:
            print('distributions',distributions)
        # Find the MLE mean (ref_mu_j) and std (ref_sigma_j) for each observable
        self.ref_mean  = np.zeros(n_observables)
        self.ref_sigma = np.zeros(n_observables)
        for j in range(n_observables):
            self.ref_mean[j] =  np.array(distributions[j]).mean()
            squared_diffs = [ (d - self.ref_mean[j])**2.0 for d in distributions[j] ]
            self.ref_sigma[j] = np.sqrt( np.array(squared_diffs).sum() / (len(distributions[j])+1.0))
            #print(self.ref_sigma[j])
        #print('ref_mean',self.ref_mean)
        #print('ref_sigma',self.ref_sigma)
        if use_global_ref_sigma == True:
            # Use the variance across all ref_sigma[j] values to calculate a single value of ref_sigma for all observables
            global_ref_sigma = ( np.array([self.ref_sigma[j]**(-2.0) for j in range(n_observables)]).mean() )**-0.5
            for j in range(n_observables):
                self.ref_sigma[j] = global_ref_sigma
                #print(self.ref_sigma[j])

        # store the ref_mean and ref_sigma information in each structure and compute/store the -log P_potential
        for s in self.ensemble:
            s[rest_index].ref_mean = self.ref_mean
            s[rest_index].ref_sigma = self.ref_sigma
            s[rest_index].compute_neglog_gaussian_ref()


    def build_exp_ref_pf(self,rest_index):
        """Calculate the MLE average PF values for restraint j across all structures,

        >>    beta_PF_j = np.array(protectionfactor_distributions[j]).sum()/(len(protectionfactor_distributions[j])+1.0)

        then use this information to compute the reference prior for each structures.
        *** VAV: NOTE that this reference potential probably should NOT be used for protection factors, PF! ***"""

        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
        #print('n_observables = ',n_observables)

        for s in self.ensemble:   # s is a list of Restraint() objects, we are considering the rest_index^th restraint
            s[rest_index].betas = []
        # for each restraint, find the average model_protectionfactor (a 6-dim array in parameter space) across all structures
        for j in range(len(s[rest_index].restraints)):
            running_total = np.zeros(self.ensemble[0][rest_index].restraints[j]['model'].shape)
            for s in self.ensemble:
                running_total += s[rest_index].restraints[j]['model']
            beta_pf_j = running_total/(len(s[rest_index].restraints)+1.0)
            for s in self.ensemble:
                s[rest_index].betas.append(beta_pf_j)
        # With the beta_PF_j values computed (and stored in each structure), now we can calculate the neglog reference potentials
        for s in self.ensemble:
            s[rest_index].compute_neglog_exp_ref_pf()


    def build_gaussian_ref_pf(self, rest_index):
        """Calculate the mean and std PF values for restraint j across all structures,
        then use this information to compute a gaussian reference prior for each structure.
        *** VAV: NOTE that this reference potential probably should NOT be used for protection factors, PF! ***"""
        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint
        #print('n_observables = ',n_observables)
        # Find the MLE mean (ref_mu_j) and std (ref_sigma_j) for each observable
        for s in self.ensemble:
            s[rest_index].ref_mean = []
            s[rest_index].ref_sigma = []
        # for each restraint, find the average model_protectionfactor (a 6-dim array in parameter space) across all structures
        for j in range(len(s[rest_index].restraints)):
            mean_PF_j  = np.zeros( self.ensemble[0][rest_index].restraints[j]['model'].shape )
            sigma_PF_j = np.zeros( self.ensemble[0][rest_index].restraints[j]['model'].shape )
            for s in self.ensemble:
                mean_PF_j += s[rest_index].restraints[j]['model']   # a 6-dim array
            mean_PF_j = mean_PF_j/(len(s[rest_index].restraints)+1.0)
            for s in self.ensemble:
                sigma_PF_j += (s[rest_index].restraints[j]['model'] - mean_PF_j)**2.0
            sigma_PF_j = np.sqrt(sigma_PF_j/(len(s[rest_index].restraints)+1.0))
            for s in self.ensemble:
                s[rest_index].ref_mean.append(mean_PF_j)
                s[rest_index].ref_sigma.append(sigma_PF_j)
        for s in self.ensemble:
            s[rest_index].compute_neglog_gaussian_ref_pf()


    def compile_nuisance_parameters(self, verbose=False):
        """Compiles arrays into a list for each nuisance parameter.

        :return np.array: numpy array with shape(n_restraint,n_para)
          [[allowed_sigma_cs_H],[allowed_sigma_noe,allowed_gamma_noe],...,[Nth_restraint]]"""

        # Generate empty lists for each restraint to fill with nuisance parameters
        nuisance_para = [ ]
        for R in self.ensemble[0]:
            # Get nuisance parameters for this specific restraint
            for para in R._nuisance_parameters:
                nuisance_para.append(np.array(getattr(R, para)))
        self.nuisance_para = np.array(nuisance_para)
#        np.save('compiled_nuisance_parameters.npy',self.nuisance_para)
        # Construct the matrix converting all values to floats for C++
        if verbose == True:
            print(self.nuisance_para)
            print('Number of Restraints = ',len(self.nuisance_para))


    def neglogP(self, new_state, parameters, parameter_indices, verbose=False):
        """Return -ln P of the current configuration.

        :param int new_state: the new conformational state from Sample()
        :param list parameters: a list of the new parameters for each of the restraints
        :param list parameter_indices: parameter indices that correspond to each restraint
        """

        s = self.ensemble[int(new_state)] # Current Structure (list of restraints)
        ln2pi = self.ln2pi                # for normalization
        _result = s[0].energy + self.logZ  # Grab the free energy of the state and normalize
        for index,R in enumerate(s):
            #print(index, R)
            _result += R.compute_neglogP(index, parameters, parameter_indices, ln2pi)
        return _result


    def sample(self, nsteps, print_freq=1000, verbose=False, debug=False):
        """Perform n number of steps (nsteps) of posterior sampling, where Monte
        Carlo moves are accepted or rejected according to Metroplis criterion.
        :param int nsteps: number of steps of sampling.
        :var int default=1000 print_freq: frequency of printing to the screen
        :var bool verbose: print statements for output
        :var bool debug: extra print statements for debugging
        See :class:`neglogP`."""

        # Generate a matrix of nuisance parameters
        self.compile_nuisance_parameters()

        # Generate a high dimentional matrix of allowed nuisance parameters
        #grid = []
        #for rest in self.nuisance_para:
        #    grid.append(np.zeros((rest.shape[0],rest.shape[0])))
        # create a list to record sampled times in each nuisance parameter space
        sampled = np.zeros(len(self.nuisance_para))

        # Generate random restraint index to initialize the sigma and gamma parameters
        self.new_rest_index = np.random.randint(len(self.ensemble[0]))

        # Initialize the state
        self.new_state = int(self.state)

        # Store a list of parameter indices for each restraint inside a list
        # parameter_indices e.g., [[161], [142], ...]
        _parameter_indices = [ ]

        # Store a list of parameters that correspond to the index inside a list
        # parameters e.g., [[1.2122652], [0.832136160], ...]
        _parameters = [ ]
        # Loop through the restraints, and get the parameters and indices
        for rest_index in range(len(self.ensemble[self.new_state])):
            Restraint = self.ensemble[self.new_state][rest_index]
            _parameter_indices.append(
                    [getattr(Restraint, para) for para in Restraint._parameter_indices])
            _parameters.append(
                    [getattr(Restraint, para) for para in Restraint._parameters])

        if verbose:
            header = """Step\t\tState\tPara Indices\t\tEnergy\t\tAcceptance (%)"""
            print(header)


        # The new parameter index to use in initial step (this is for restraints with more than one nuisance parameters)
        #self.new_para_index = np.random.randint(
        #        len(_parameter_indices[self.new_rest_index]) )

        # Create separate accepted ratio recorder list
        n_para = 1
        for para in _parameters:    # restraint
            for in_para in para:    # nuisance parameters
                n_para += 1
        sep_accepted = np.zeros(n_para)   # all nuisance paramters + state (n_para starts from 1 not 0)

        for step in range(nsteps):

            # Redefine based upon acceptance (Metroplis criterion)
            new_state = self.new_state

            # parameter_indices e.g. [[161], [142]]
            parameter_indices = _parameter_indices

            """ the point of the following part is to convert
            the original parameters in the format of list of lists (e.g. [[1],[2,3]])
            to a 1D list (e.g. [1,2,3] because the allowed_parameters (self.nuisance_para)
            is in the format of [a,b,c] where a,b,c represent parameters
            (which is different from parameter_indices).
            So the logic is to convert from list of lists to a 1D list
            and track the original index of each parameters to convert it back to the list of lists later.
            (The reason for that is because the way we coded up neglogP function requires that format).
            The code will randomly pick up one observable space (including state space) to sample
            and then propose a jump along each parameter space associated with that observable.
            Once we have the new index and parameter of that observable,
            we convert all parameters/indices back to the original format (list of lists)
            and feed them to neglogP function for energy calculation.
            I'm sure this part can be improved and I suggest people who are going to work on this
            read the code carefully and make sure you fully understand what is going on here
            and come up with your own way to make it better.
            This is the core part of BICePs so make sure you know what you are doing. --Yunhui Ge 03/2020)
            """

            # make a temporary list of indices
            temp_parameter_indices = []
            original_index =[]   # keep tracking the original index of the parameters
            for ind in range(len(parameter_indices)):
                for in_ind in parameter_indices[ind]:
                    temp_parameter_indices.append(in_ind)
                    original_index.append(ind)
            original_index = np.array(original_index)
            #print('original_index',original_index)
            # parameters e.g. [[1.2122652], [0.832136160]]
            parameters = _parameters
            # make a temporary list of parameters
            temp_parameters = []
            for para in parameters:
                for in_para in para:
                    temp_parameters.append(in_para)

            # RAND = generalized probability of taking a step in restraint space given the total number of restraints.
            RAND = 1. - 1./(len(parameter_indices) + 1.)   # 1. is the state
            #print('RAND',RAND)
            # randomly pick up one observable to sample
            to_sample_ind = np.random.randint(len(parameter_indices))  # together with RAND, this will make sure all observables and the state space will share the same probability to propose a MCMC movement
            #print('to_sample_ind',to_sample_ind)
            sample_ind = np.where(original_index==to_sample_ind)[0]   # the ind in the list of parameters (later used for self.nuisance_para)
            #print('sample_ind',sample_ind)
            # find corresponding nuisance parameters
            allowed_parameters = []
            for para_ind in sample_ind:
                allowed_parameters.append(self.nuisance_para[para_ind])
            #print('allowed_parameters',allowed_parameters)
            # pick up the index of parameters associated with the observable to sample
            index = []
            for para_ind in sample_ind:
                index.append(temp_parameter_indices[para_ind])
            #print('index',index)
            #ind1 = index

            # rolling a dice
            dice = np.random.random()
            #print('rolling dice', dice)
            #if np.random.random() < RAND:

            if dice < RAND:
                ## Take a random step in the space of specific parameter
                actual_sample_ind = sample_ind  # actual parameters being sampled
                #print('actual_sample_ind',actual_sample_ind)
                # Shift the index by +1, 0 or -1
                temp_index = []
                for ind in range(len(index)):
                    temp_index.append(index[ind] + (np.random.randint(3)-1))
                #print('temp_index',temp_index)
                #index += (np.random.randint(3)-1)
#                index = np.random.randint(len(nuisance_para))
                # Make sure the index doesn't fall out of the boundry of the allowed parameters
                temp_index2 = []
                for ind in range(len(temp_index)):
                    temp_index2.append(temp_index[ind]%len(allowed_parameters[ind]))
                #print('temp_index2',temp_index2)
                #index = index%(len(allowed_parameters))
                #ind2 = index
                ## Temporary replacement until satisfied by Metroplis criterion
                # Replace the old parameter with the new parameter
                for ind in range(len(sample_ind)):
                    temp_parameters[sample_ind[ind]] = allowed_parameters[ind][temp_index2[ind]]
                    temp_parameter_indices[sample_ind[ind]] = temp_index2[ind]
                #print('temp_parameters',temp_parameters)
                #print('temp_parameter_indices',temp_parameter_indices)
                #temp_parameters[to_sample_ind] = allowed_parameters[index]
                #temp_parameter_indices[to_sample_ind] = index

            else:
                ## Take a random step in state space
                new_state = np.random.randint(self.nstates)
                actual_sample_ind = [len(temp_parameters)]  # actual parameters being sampled, if it's state then it's the last one in the list
            if debug:
                print('*****************************************')
                #print('new_rest_index ', new_rest_index )
                #print('new_para_index ', new_para_index )
                print('new_state ', new_state )
                #print('new_sigma ', temp_parameters[to_sample_ind] )
                #print('new_sigma_index ', temp_parameter_indices[to_sample_ind] )
                #print('new_allowed_parameters ', allowed_parameters )
                print('parameter_indices ', temp_parameter_indices )
                print('parameters ',temp_parameters)
                print('*****************************************')

            # recreate a list with the same shape of the original list required by the neglogP function
            new_parameters=[[] for l in range(len(parameter_indices))]
            new_parameter_indices=[[] for l in range(len(parameter_indices))]
            for m in range(len(original_index)):
                new_parameters[original_index[m]].append(temp_parameters[m])
                new_parameter_indices[original_index[m]].append(temp_parameter_indices[m])
            # record which nuisance parameters space is sampled
            sampled[to_sample_ind] += 1.0

            # Compute new "energy"
            new_E = self.neglogP(new_state, new_parameters, new_parameter_indices, verbose=False)

            # Accept or reject the MC move according to Metroplis criterion
            accept = False
#            print('new_E',new_E,'self.E',self.E)
            if new_E < self.E:
                accept = True

            else:
                if np.random.random() < np.exp( self.E - new_E ):
                    accept = True
            # Update parameters based upon acceptance (Metroplis criterion)
            if accept:
                self.E = new_E
                self.new_state = new_state
                _parameter_indices = new_parameter_indices
                _parameters = new_parameters
                for ind in actual_sample_ind:
                    sep_accepted[ind] += 1.0
                #sep_accepted[actual_sample_ind] += 1.0  # keep recording accepted step based on which parameters sampled
                self.accepted += 1.0
                #if actual_sample_ind == len(temp_parameters):
                #    grid[to_sample_ind][ind1,ind1] += 1.0
                #else:
                #    grid[to_sample_ind][ind1,ind2] += 1.0
            self.total += 1.0

            if debug:
                if accept:
                    print('*****************************************')
                    print('self.E', self.E)
                    print('self.new_state ', self.new_state )
                    print('self.new_rest_index ', self.new_rest_index )
                    #print('self.new_para_index ', new_para_index)
                    print('self.accepted', self.accepted)
                    print('*****************************************')

            #NOTE: There will need to be additional parameters here for protection factor.


            self.traj.state_counts[int(self.new_state)] += 1
            self.traj.state_trace.append(int(self.new_state))

            # Store the counts of sampled sigma along the trajectory
            for i in range(len(np.concatenate(_parameter_indices))):
                self.traj.sampled_parameters[i][np.concatenate(_parameter_indices)[i]] += 1

            # Store trajectory samples
            temp=[[] for i in range(len(_parameter_indices))]
            for i in range(len(_parameter_indices)):
                for j in _parameter_indices[i]:
                    temp[i].append(int(j))

            # Store trajectory samples
            if step%self.traj_every == 0:
                self.traj.trajectory.append( [int(step), float(self.E),
                    int(accept), int(self.new_state), list(temp)])
                self.traj.traces.append(np.concatenate(_parameters))

            if verbose:
                if step%print_freq == 0:
                    output = """%i\t\t%i\t%s\t%.3f\t\t%.2f"""%(step,self.new_state,
                            _parameter_indices,self.E,self.accepted/self.total*100.)
                    print(output)

        print('\nAccepted %s %% \n'%(self.accepted/self.total*100.))
        print('\nAccepted [ ...Nuisance paramters..., state] %')
        print('Accepted %s %% \n'%(sep_accepted/self.total*100.))
        self.traj.sep_accept.append(sep_accepted/self.total*100.)    # separate accepted ratio
        self.traj.sep_accept.append(self.accepted/self.total*100.)   # the total accepted ratio
        #for g in range(len(grid)):
        #    self.traj.grid.append(grid[g]/sampled[g]*100.)




class PosteriorSamplingTrajectory(object):
    """A container class to store and perform operations on the trajectories of
    sampling runs."""

    def __init__(self, ensemble):
        """Initialize the PosteriorSamplingTrajectory container class.

        :param list ensemble: """

        self.ensemble = ensemble
        self.nstates = len(self.ensemble)
        self.state_counts = np.ones(self.nstates)  # add a pseudo-count to avoid log(0) errors

        # Lists for each restraint inside a list
        self.sampled_parameters = []
        self.allowed_parameters = []
        self.ref = [ []  for i in range(len(ensemble[0]))]  # parameters of reference potentials
        self.model = [ [] for i in range(len(ensemble[0]))]  # restraints model data
        self.sep_accept = []     # separate accepted ratio
        #self.grid = []   # for acceptance ratio plot
        self.state_trace = []
        s = self.ensemble[0]

        for rest_index in range(len(s)):
            # TODO: Is there a better wat to call on these variables?
            # right now it is a list of strings
            nuisance_parameters = getattr(s[rest_index], "_nuisance_parameters")
            for para in nuisance_parameters:
                self.allowed_parameters.append(getattr(s[rest_index], para))
                self.sampled_parameters.append(np.zeros(len(getattr(s[rest_index], para))))

        # Generate a list of the names of the parameter indices for the traj header
        parameter_indices = []
        for rest_index in range(len(s)):
            # TODO: Is there a better wat to call on these variables?
            # right now it is a list of strings
            parameter_indices.append( getattr(s[rest_index], '_parameter_indices') )

        self.rest_type = []
        for rest_index in range(len(s)):
            # TODO: Is there a better wat to call on these variables?
            for rest_type in getattr(s[rest_index], '_rest_type'):
                self.rest_type.append(rest_type)

        self.trajectory_headers = ["step", "E", "accept", "state",
                "para_index = %s"%parameter_indices]

        self.trajectory = []
        self.traces = []
        self.results = {}

    def process_results(self, outfilename='traj.npz'):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables."""

        # Store the name of the restraints in a list corresponding to the correct order
        saving = ['rest_type','trajectory_headers','trajectory','sep_accept',
                'allowed_parameters','sampled_parameters','model','ref','traces','state_trace']

        for rest_index in range(len(self.ensemble[0])):
            n_observables  = self.ensemble[0][rest_index].n
            for n in range(n_observables):
                model = []
                for s in range(len(self.ensemble)):
                    model.append(self.ensemble[s][rest_index].restraints[n]['model'])
                self.model[rest_index].append(model)

        #TODO: Check to make sure that there hasn't been an update in Py3
        # that will allow datatype convervation in the method `getattr()`

        self.results['rest_type'] = self.rest_type
        self.results['trajectory_headers'] = self.trajectory_headers
        self.results['trajectory'] = self.trajectory
        self.results['sep_accept'] = self.sep_accept
        #self.results['grid'] = self.grid
        self.results['allowed_parameters'] = self.allowed_parameters
        self.results['sampled_parameters'] = self.sampled_parameters
        self.results['model'] = self.model
        self.results['ref'] = self.ref
        self.results['traces'] = self.traces
        self.results['state_trace'] = self.state_trace

        self.write(outfilename, self.results)


    def write(self, outfilename='traj.npz', *args, **kwds): # new
        """Writes a compact file of several arrays into binary format.
        Standardized: Yes ; Binary: Yes; Human Readable: No;

        :param str outfilename: name of the output file
        :return: numpy compressed filetype
        """
        np.savez_compressed(outfilename, *args, **kwds)







