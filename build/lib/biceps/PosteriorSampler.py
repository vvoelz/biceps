# -*- coding: utf-8 -*-
import numpy as np
import inspect, time
from .KarplusRelation import *     # Returns J-coupling values from dihedral angles
from .Restraint import *
from .toolbox import *
from tqdm import tqdm # progress bar

class PosteriorSampler(object):

    def __init__(self, ensemble, freq_write_traj=100., freq_save_traj=100., verbose=False):
        """A class to perform posterior sampling of conformational populations.

        Args:
            ensemble(object): a :attr:`biceps.Ensemble` object
            freq_write_traj(int): the frequency (in steps) to write the MCMC trajectory
            freq_print(int): the frequency (in steps) to print status
            freq_save_traj(int): the frequency (in steps) to store the MCMC trajectory
        """

        self.lam = ensemble.lam
        self.ensemble = ensemble.to_list() # Allow the ensemble to pass through the class
        self.nreplicas = 1
        self.write_traj = freq_write_traj # Step frequencies to write trajectory info
        self.traj_every = freq_save_traj # Frequency of storing trajectory samples
        self.nstates = len(self.ensemble) # Ensemble is a list of Restraint objects
        # The initial state of the structural ensemble we're sampling from
        self.state = 0    # index in the ensemble
        self.state = np.random.randint(low=0, high=self.nstates, size=self.nreplicas)
        self.E = 1.0e99   # initial energy
        self.accepted = 0
        self.total = 0
        # keep track of what we sampled in a trajectory
        self.traj = PosteriorSamplingTrajectory(ensemble=ensemble, sampler=self, nreplicas=self.nreplicas)
        # for each Restraint, calculate global reference potential parameters
        # ..by looking across all structures
        # TODO: can't this be more general?!?
        for i,R in enumerate(self.ensemble[0]):
            if R.ref == "uniform":
                self.traj.ref[i].append('Nan')
                pass
            elif R.ref == 'exponential':
                if hasattr(R, 'precomputed'):
                    if not R.precomputed:
                        self.build_exp_ref_pf(i)
                else:
                    self.build_exp_ref(i)
                self.traj.ref[i].append(R.betas)
            elif R.ref == 'gaussian':
                if hasattr(R, 'precomputed'):
                    if not R.precomputed:
                        self.build_gaussian_ref_pf(i, use_global_ref_sigma=R.use_global_ref_sigma)
                else:
                    self.build_gaussian_ref(i, use_global_ref_sigma=R.use_global_ref_sigma)
                self.traj.ref[i].append(R.ref_mean)
                self.traj.ref[i].append(R.ref_sigma)
            else:
                raise ValueError('Please choose a reference potential of the following:\n \
                    {%s,%s,%s}'%('uniform','exponential','gaussian'))
        # Compute ref state logZ for the free energies to normalize.
        self.compute_logZ()
        self.verbose = verbose

    def compute_logZ(self):
        """Compute reference state logZ for the free energies to normalize."""

        Z = 0.0
        for s in self.ensemble:
            #Z +=  np.exp( -np.array(s[0].energy, dtype=np.float128) )
            Z +=  np.exp( -np.array(s[0].energy, dtype=np.float64) )
        self.logZ = np.log(Z)


    def build_exp_ref(self, rest_index, verbose=False):
        """Looks at each structure to find the average observables
        :math:`<r_{j}>`, then stores the reference potential info for each
        :attr:`biceps.Restraint.Restraint` of this type for each structure.

        ``beta_j = np.array(distributions[j]).sum()/(len(distributions[j])+1.0)``

        Args:
            rest_index(int): restraint index
        """

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
        """Looks at all the structures to find the mean (:math:`\\mu`) and std
        (:math:`\\sigma`) of observables r_j then store this reference potential
        info for all restraints of this type for each structure.

        Args:
            rest_index(int): restraint index
            use_global_ref_sigma(bool):
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

        ``beta_PF_j = np.array(protectionfactor_distributions[j]).sum()/(len(protectionfactor_distributions[j])+1.0)``

        then use this information to compute the reference prior for each structures.

        .. tip:: **(not required)** an additional method specific for protection factor
        """

        # collect distributions of observables r_j across all structures
        n_observables  = self.ensemble[0][rest_index].n  # the number of (model,exp) data values in this restraint

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

        .. tip:: **(not required)** an additional method specific for protection factor
        """

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
        """Compiles numpy arrays of allowed parameters for each nuisance parameter.

        :rtype np.ndarray: with shape(n_restraint,n_para)
        """

        # Generate empty lists for each restraint to fill with nuisance parameters
        nuisance_para = [ ]
        for R in self.ensemble[0]:
            keys = R.__dict__.keys() # all attributes of the Child Restraint class
            for j in [key for key in keys if "allowed_" in key]: # get the allowed parameters
                nuisance_para.append(np.array(getattr(R, j)))
        self.nuisance_para = np.array(nuisance_para, dtype=object)
#        np.save('compiled_nuisance_parameters.npy',self.nuisance_para)
        # Construct the matrix converting all values to floats for C++
        if verbose == True:
            print(self.nuisance_para)
            print('Number of Restraints = ',len(self.nuisance_para))
        return self.nuisance_para


    def neglogP(self, states, parameters, parameter_indices):
        """Return -ln P of the current configuration.

        Args:
            state(list): the new conformational state being sampled in :attr:`PosteriorSampler.sample`
            parameters(list): a list of the new parameters for each of the restraints
            parameter_indices(list): parameter indices that correspond to each restraint
        """

        result = 0
        for state in states:
            s = self.ensemble[int(state)] # Current Structure (list of restraints)
            result += s[0].energy + self.logZ  # Grab the free energy of the state and normalize
            for i,R in enumerate(s):
                result += R.compute_neglogP(parameters[i], parameter_indices[i], s[i].sse)
        return result


    def sample(self, nsteps, burn=0, print_freq=1000, verbose=False, progress=True):
        """Perform n number of steps (nsteps) of posterior sampling, where Monte
        Carlo moves are accepted or rejected according to Metroplis criterion.
        Energies are computed via :class:`neglogP`.

        Args:
            nsteps(int): the number of steps of sampling
            burn(int): the number of steps to burn
            print_freq(int): the frequency of printing to the screen
            verbose(bool): control over verbosity

        .. tip::
            Set `verbose=False` when using multiprocessing.
        """

        # Generate a matrix of nuisance parameters
        allowed = self.compile_nuisance_parameters()
        # Store a list of nuisance parameters for each restraint
        self.rest_type = []
        # Store a list of parameter indices for each restraint inside a list
        self.indices = [] # e.g., [161, 142, ...]
        # Loop through the restraints, and get the parameters and indices
        rest_index = []
        #NOTE: using information from first state
        for i,R in enumerate(self.ensemble[self.state[0]]):
            keys = R.__dict__.keys() # all attributes of the Child Restraint class
            for j in [key for key in keys if "index" in key]: # get the parameter indices
                self.indices.append(getattr(R, j))
            for j in [key.split("_")[-1] for key in keys if "allowed_" in key]: #
                self.rest_type.append(str(j)+"_"+str(R.__repr__).split("_")[-1].split()[0])
                rest_index.append(i)
        if verbose:
            header = """Step\t\tState\tPara Indices\t\tAvg Energy\tAcceptance (%)"""
            print(header)
        # Create separate accepted ratio recorder list
        n_rest = max(rest_index)+1
        sep_accepted = np.zeros(len(self.indices)+1) # all nuisance paramters + state (n_para starts from 1 not 0)
        step=0
        start = time.time()

        if (not verbose) and progress: pbar = tqdm(total=nsteps+burn)
        while step < nsteps+burn:
            # Redefine based upon acceptance (Metroplis criterion)
            state, E = self.state.copy(), self.E
            indices = self.indices.copy() # e.g. [161, 142]
            #values = self.values # e.g. [1.2122652, 0.832136160]
            # All sample-space will share the same probability to be sampled
            RAND = 1. - 1./(n_rest + 1.)   # + 1. is the term to include state-space
            dice = np.random.random() # rolling the dice
            if dice < RAND: # Take a random step in Restraint space
                ind = []
                # Make sure the index doesn't fall out of the boundry of the allowed values
                for k in np.where(np.array(rest_index)==np.random.randint(n_rest))[0]:
                    indices[k] = (indices[k]+(np.random.randint(3)-1))%len(allowed[k])
                    ind.append(k)
            else: ## Take a random step in state space
                state = np.random.randint(low=0, high=self.nstates, size=self.nreplicas)
                ind = [len(indices)]
            # values e.g., [1.2122652, 0.832136160, ...]
            values = [allowed[i][indices[i]] for i in range(len(indices))]
            # Convert the list of indices and values into a list of list for each restraint
            sep_indices = [[] for i in range(n_rest)]
            sep_values = [[] for i in range(n_rest)]
            for n,m in enumerate(rest_index):
                sep_indices[m].append(indices[n])
                sep_values[m].append(values[n])
            #print(f"sep_indices: {sep_indices}")
            # Compute new "energy"
            E = self.neglogP(state, sep_values, sep_indices)
            # Accept or reject the MC move according to Metroplis criterion
            self.accept = False
            if E < self.E:
                self.accept = True
            else:
                if np.random.random() < np.exp( self.E - E ):
                    self.accept = True

            # Update values based upon acceptance (Metroplis criterion)
            if self.accept:
                self.E = E
                self.state = state.copy()
                self.indices = indices.copy()
                self.values = values.copy()
                for k in ind:
                    sep_accepted[k] += 1.0
                self.accepted += 1.0
            self.total += 1.0

            if (step >= burn):
                _step = step-burn
                if (not verbose) and progress: pbar.update(1)
                # Store sampled states along trajectory
                for i in range(len(self.state)):
                    self.traj.state_counts[int(self.state[i])] += 1
                    self.traj.state_trace.append(int(self.state[i]))
                # Store the counts of sampled sigma along the trajectory
                for i in range(len(self.indices)):
                    self.traj.sampled_parameters[i][self.indices[i]] += 1
                # Store trajectory samples
                temp=[[] for i in range(n_rest)]
                for n,m in enumerate(rest_index):
                    temp[m].append(self.indices[n])
                # Store trajectory samples
                if (_step%self.traj_every == 0):
                    self.traj.trajectory.append( [int(_step), float(self.E),
                        int(self.accept), list(self.state.copy()), list(temp.copy())])
                        #int(self.accept), int(self.state), list(temp.copy())])
                    self.traj.traces.append(self.values.copy())

                if verbose:

                    if _step%print_freq == 0:
                        output = """%i\t\t%s\t%s\t\t%.3f\t\t%.2f\t%s"""%(_step, self.state,
                                self.indices, self.E/self.nreplicas, self.accepted/self.total*100., self.accept)
                        print(output)
            step += 1
        restraints = list(dict.fromkeys([r.split("_")[-1] for r in self.rest_type]))
        if not verbose: pbar.close()

        print('\nAccepted %s %% \n'%(self.accepted/self.total*100.))
        print('\nAccepted [ ...Nuisance paramters..., state] %')
        print('Accepted %s %% \n'%(sep_accepted/self.total*100.))
        self.traj.sep_accept.append(sep_accepted/self.total*100.)    # separate accepted ratio
        self.traj.sep_accept.append(self.accepted/self.total*100.)   # the total accepted ratio



class PosteriorSamplingTrajectory(object):
    def __init__(self, ensemble, sampler, nreplicas):
        """A container class to store and perform operations on the trajectories of
        sampling runs.

        Args:
            ensemble(list): ensemble of :attr:`biceps.Restraint.Restraint` objects
            nreplicas(int): number of replicas
        """

        self.lam = ensemble.lam
        self.sampler = sampler
        self.ensemble = ensemble.to_list() # Allow the ensemble to pass through the class
        self.nreplicas = nreplicas
        self.nstates = len(self.ensemble)
        self.state_counts = np.ones(self.nstates)  # add a pseudo-count to avoid log(0) errors

        # Lists for each restraint inside a list
        self.sampled_parameters = []
        self.allowed_parameters = []
        self.ref = [ []  for i in range(len(self.ensemble[0]))]  # parameters of reference potentials
        self.model = [ [] for i in range(len(self.ensemble[0]))]  # restraints model data
        self.sep_accept = []     # separate accepted ratio
        self.state_trace = []
        s = self.ensemble[0]
        # Generate a list of the names of the parameter indices for the traj header
        parameter_indices = []
        self.rest_type = []
        for i,R in enumerate(s):
            keys = R.__dict__.keys() # all attributes of the Child Restraint class
            for j in [key for key in keys if "allowed_" in key]: # get the allowed parameters
                self.allowed_parameters.append(getattr(R, j))
                self.sampled_parameters.append(np.zeros(len(getattr(R, j))))
            for j in [key for key in keys if "index" in key]: # get the parameter indices
                parameter_indices.append(getattr(R, j))
            for j in [key.split("_")[-1] for key in keys if "allowed_" in key]: #
                self.rest_type.append(str(j)+"_"+str(R.__repr__).split("_")[-1].split()[0])

        self.trajectory_headers = ["step", "E", "accept", "state",
                "para_index = %s"%parameter_indices]
        self.trajectory = []
        self.traces = []
        self.results = {}

    def process_results(self, filename=None):
        """Process the trajectory, computing sampling statistics,
        ensemble-average NMR observables.

        Benefits of using Numpy Z compression (npz) formatting:
        1) Standardized Python library (NumPy), 2) writes a compact file
        of several arrays into binary format and 3) significantly smaller
        size over many other formats.

        Args:
            filename(str): relative path and filename for MCMC trajectory

        .. tip::

            It is possible to convert the trajectory file to a Pandas DataFrame
            (pickle file) with the following: :attr:`biceps.toolbox.npz_to_DataFrame`

        """

        if filename == None: filename = f"traj_lambda{self.lam}.npz"

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
        self.results['allowed_parameters'] = self.allowed_parameters
        self.results['sampled_parameters'] = self.sampled_parameters
        self.results['model'] = self.model
        self.results['ref'] = self.ref
        self.results['traces'] = self.traces
        self.results['state_trace'] = self.state_trace

        self.write(filename, self.results)
        # Save Sampler object
        save_object(self.sampler, filename.replace(".npz",".pkl"))

        #TODO: Return a Pandas Dataframe of the results to be passed into
        # Analysis so time isn't wasted loading in long trajectories
        #return self.results
        #return pd.DataFrame(self.results)


    def write(self, filename='traj.npz', *args, **kwds):
        """Writes a compact file of several arrays into binary format.
        Standardized: Yes ; Binary: Yes; Human Readable: No;

        Args:
            filename(str): path and filename of output MCMC trajectory

        :rtype: npz (numpy compressed file)
        """

        np.savez_compressed(filename, *args, **kwds)







