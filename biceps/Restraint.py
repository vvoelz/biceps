# -*- coding: utf-8 -*-

import os, sys, inspect, time, copy
import numpy as np
import pandas as pd
import mdtraj as md
import biceps
from biceps.KarplusRelation import * # Returns J-coupling values from dihedral angles

def print_attr(obj):
    for p,s in enumerate(obj):
        attr = dir(s)
        for at in attr:
            if not at.startswith('__'):
                print(at)



def get_restraint_options(input_data=None):
    """Return a Pandas DataFrame of all the parameters for each of the restraint
    classes. If input_data is provided, then the DataFrame returned will only
    contain restraint parameters that corresponds to the data.  When providing
    input_data, the order of restraints (rows) corresponds to the order of
    the data.

    NOTE: If you want to use these default options, then use `options.to_dict('records')`

    Args:
        input_data(list) - ordered list of data for each state

    Returns:
        options(pd.DataFrame)
    """

    options = []
    current_module = sys.modules[__name__]
    # Pick the Restraint class upon file extension
    restraints = biceps.toolbox.list_possible_restraints()

    # NOTE: if input_data is given, then remove information regarding all other restraints
    if input_data != None:
        scheme = biceps.toolbox.list_res(input_data)
        #extensions = [ res.split("_")[-1] for res in scheme ]
        extensions = scheme
        _restraints = []
        for rest in scheme:
            for restraint in restraints:
                if restraint.split("_")[-1] in rest:
                    _restraints.append(restraint)
        restraints = _restraints

    keys_to_remove = ["data", "energy", "self", "verbose"]
    for k,restraint in enumerate(restraints):
        _options = {}
        R = getattr(current_module, "%s"%(restraint))

        # Get all the arguments for the Parent Restraint Class if given
        args1 = inspect.getfullargspec(R)
        args2 = inspect.getfullargspec(R.init_restraint)

        sig1 = inspect.signature(R)
        for key in sig1.parameters.keys():
            value = sig1.parameters[key].default
            if value == inspect._empty:
                _options[key] = []
            else:
                _options[key] = [value]

        sig2 = inspect.signature(R.init_restraint)
        for key in sig2.parameters.keys():
            value = sig2.parameters[key].default
            if value == inspect._empty:
                _options[key] = [np.nan]
            else:
                _options[key] = [value]
        # remove all args that user doesn't control
        [_options.pop(key, None) for key in keys_to_remove]
        # if given input_data, replace the extension with suggested
        if input_data != None: _options["extension"] = [extensions[k]]
        # construct a dataframe for each restraint type
        df = pd.DataFrame(_options, index=[R.__name__])
        options.append(df.to_dict('records')[0])
    return options


class ExpandedEnsemble:
    def __init__(self, energies, lambda_values=None, xi_values=None, expanded_values=None,
             diff_energies=None, diff2_energies=None):
        """ExpandedEnsemble class constructs a list of ensembles from specified
        lambda_values or expanded_values.
        """

        self.energies = energies
        self.ensembles = []
        self.expanded_values = []

        # FIXME: doesn't do anything with xi_values if the energies are multidimensional
        if energies.ndim == 2:
            if isinstance(expanded_values, (np.ndarray, list)):
                self.expanded_values = expanded_values
                self.xi_values = []
                self.lambda_values = []
                for i,vals in enumerate(self.expanded_values):
                    lam, xi = vals
                    self.xi_values.append(xi)
                    self.lambda_values.append(lam)
                    self.ensembles.append(Ensemble(lam, xi, self.energies[i], diff_energies, diff2_energies))
            else:
                self.xi_values = []
                self.lambda_values = lambda_values
                for i,lam in enumerate(lambda_values):
                    self.xi_values.append(1.0)
                    self.ensembles.append(Ensemble(lam, self.xi_values[i], self.energies[i], diff_energies, diff2_energies))
                    self.expanded_values.append((lam, self.xi_values[i]))
            return

        if isinstance(expanded_values, (np.ndarray, list)):
            # user provided ordered lambda and xi values for each trajectory
            self.expanded_values = expanded_values
            self.xi_values = []
            self.lambda_values = []
            for i,vals in enumerate(self.expanded_values):
                lam, xi = vals
                self.xi_values.append(xi)
                self.lambda_values.append(lam)
                self.ensembles.append(Ensemble(lam, xi, self.energies, diff_energies, diff2_energies))
            return

        if xi_values != None:
            # user provided xi values for lambda == 0.0
            self.xi_values = [float("%0.2f"%xi) for xi in xi_values]
            if lambda_values:
                self.lambda_values = [float("%0.2f"%lam) for lam in lambda_values]
            else:
                raise ValueError("lambda_values be a list or numpy array with type of 'float'")
            self.expanded_values = []
            for i,lam in enumerate(self.lambda_values):
                # NOTE: Xi values for only when lam is 0
                if float(lam) == 0.0:
                    for j, xi in enumerate(self.xi_values):
                        self.expanded_values.append((lam, xi))
                else:
                    self.expanded_values.append((lam, 1.0))
            for i,vals in enumerate(self.expanded_values):
                lam, xi = vals
                self.ensembles.append(Ensemble(lam, xi, self.energies, diff_energies, diff2_energies))
            return

        else:
            # your standard lambda valued trajectories
            self.expanded_values = []
            self.xi_values = []
            self.lambda_values = []
            xi = 1.0
            for i,lam in enumerate(lambda_values):
                self.xi_values.append(xi)
                self.expanded_values.append((lam, xi))
                self.lambda_values.append(lam)
                self.ensembles.append(Ensemble(lam, xi, self.energies, diff_energies, diff2_energies))
            return


    def initialize_restraints(self, input_data, options, verbose=False):
        """Initialize the restraints for a single ensemble, then make a copy of
        the ensemble changing only the energies for the other ensembles.
        """

        stime = time.time()
        self.options = options
        self.ensembles[0].initialize_restraints(input_data, options)

        if len(self.expanded_values) > 1:
            for i,exp_vals in enumerate(list(self.expanded_values[1:])):
                lam,xi = exp_vals
                i += 1
                if self.energies.ndim == 2:
                    energies = self.energies[i]
                else:
                    energies = self.energies*lam
                # must deepcopy the object so you don't get a reference
                self.ensembles[i] = copy.deepcopy(self.ensembles[0])
                self.ensembles[i].lam = lam
                self.ensembles[i].xi = xi
                self.ensembles[i].energies = energies
                for s,energy in enumerate(energies):
                    for r in range(len(self.ensembles[i].ensemble[s])):
                        self.ensembles[i].ensemble[s][r].energy = energy
#        self.initialize_fwd_model_derivatives()
        total_time = time.time() - stime
        if verbose: print(f"Time to initalize restraints: {total_time:.2f}s")


    def to_list(self):
        return [ensemble.to_list() for ensemble in self.ensembles]



    def update_forward_model(self, model_data, restraint_index):
        """Function for updating the forward model of the ensemble object."""

        for l in range(len(self.expanded_values)): # thermodynamic ensembles
            for s in range(len(self.ensembles[l].ensemble)): # conformational states
                for r in range(len(self.ensembles[l].ensemble[s])): # data restraint types
                    if r != restraint_index: continue
                    for j in range(len(self.ensembles[l].ensemble[s][r].restraints)): # data points (observables)
                        self.ensembles[l].ensemble[s][r].restraints[j]["model"] = model_data[s][j]
                        #self.ensembles[l].ensemble[s][r].restraints[j]["diff model"] =
                        #self.ensembles[l].ensemble[s][r].restraints[j]["diff2 model"] =
                    #exit()

#    def initialize_fwd_model(self, init_paras, x, indices, min_max_paras=None, parameter_priors=None, **kwargs):
#        '''
#        # Attach the following to the `ensemble` object
#        # Attach detailed model parameters and restraint specifics to the ensemble
#        # 1. The forward model parameters for each ensemble
#        self.fwd_model_parameters = fwd_model_paras # shape: (nchains, K, Np)
#        # 2. The indices belonging to the J-coupling restraints
#        self.fmo_restraint_indices = J_indices
#        # 3. The phi-angles for each state
#        # This is a nested list structure where each element is
#        # indexed first by K, then by nstates, and finally by Nd, which varies for each k
#        self.phi_angles = phi_angles # sequence: [K][nstates][Nd]
#        # 4. The phase shifts for each J-coupling type
#        self.phase_shifts = phi0 # shape: (K,)
#        '''
#
#        before = copy.deepcopy(self.__dict__)
#        self.fwd_model_parameters = init_paras
#        self.fmo_restraint_indices = indices
#        self.phi_angles = x
#        self.phase_shifts = kwargs["phi0"]
#        if min_max_paras is None:
#            self.min_max_fwd_model_parameters = np.zeros((len(self.fwd_model_parameters[0][0]), 2))
#            self.min_max_fwd_model_parameters[:,0] -= 10
#            self.min_max_fwd_model_parameters[:,1] += 10
#        else:
#            self.min_max_fwd_model_parameters = min_max_paras
#
#        if parameter_priors is None:
#            # self.fmp_prior_models has shape (Nk, Np)
#            self.fmp_prior_models = np.array([
#                ["uniform" for i in range(len(self.fwd_model_parameters[0][0]))]
#                    for k in range(len(self.fwd_model_parameters[0]))], dtype=str)
#        else:
#            self.fmp_prior_models = parameter_priors
#
#        self.fmp_prior_mus = np.zeros(self.fwd_model_parameters.shape[1:])
#
#        after = self.__dict__
#        self.fwd_model_attrs = {k: after[k] for k in after if k not in before or after[k] != before[k]}




    def initialize_prior_model(self, model, init_paras, min_max_paras=None,
            parameter_priors=None, parameter_prior_sigmas=None, parameter_prior_mus=None, **kwargs):
        '''
        # Attach the following to the `ensemble` object
        # Attach detailed model parameters and restraint specifics to the ensemble
        '''

        before = copy.deepcopy(self.__dict__)

        self.prior_model = model
        self.prior_model_parameters = init_paras

        # Handling default values for min_max_paras
        if min_max_paras is None:
            self.min_max_prior_model_parameters = np.zeros((len(init_paras), 2))
            self.min_max_prior_model_parameters[:, 0] -= 100
            self.min_max_prior_model_parameters[:, 1] += 100
        else:
            self.min_max_prior_model_parameters = min_max_paras

        # Handling default values for parameter_priors
        if parameter_priors is None:
            #self.pmp_prior_models = np.array(["uniform"]*len(init_paras), dtype=str)
            self.pmp_prior_models = np.array(["Gaussian"]*len(init_paras), dtype=str)
        else:
            self.pmp_prior_models = parameter_priors

        if parameter_prior_sigmas is None:
            self.pmp_prior_sigmas = np.ones(np.array(init_paras).shape)*2
        else:
            self.pmp_prior_sigmas = parameter_prior_sigmas

        if parameter_prior_mus is None:
            self.pmp_prior_mus = np.array(self.prior_model_parameters)
        else:
            self.pmp_prior_mus = parameter_prior_mus


        #self.pmp_prior_mus = np.copy(np.array(self.prior_model_parameters))


        after = self.__dict__
        self.prior_model_attrs = {
            k: after[k] for k in after
            if k not in before or not np.array_equal(after[k], before[k])
        }




    def initialize_fwd_model(self, init_paras, x, indices, min_max_paras=None,
            parameter_priors=None, parameter_prior_sigmas=None, parameter_prior_mus=None, **kwargs):
        '''
        # Attach the following to the `ensemble` object
        # Attach detailed model parameters and restraint specifics to the ensemble
        # 1. The forward model parameters for each ensemble
        self.fwd_model_parameters = fwd_model_paras # shape: (nchains, K, Np)
        # 2. The indices belonging to the J-coupling restraints
        self.fmo_restraint_indices = J_indices
        # 3. The phi-angles for each state
        # This is a nested list structure where each element is
        # indexed first by K, then by nstates, and finally by Nd, which varies for each k
        self.phi_angles = phi_angles # sequence: [K][nstates][Nd]
        # 4. The phase shifts for each J-coupling type
        self.phase_shifts = phi0 # shape: (K,)
        '''

        before = copy.deepcopy(self.__dict__)

        # Setting attributes, assuming the inputs are handled correctly
        self.fwd_model_parameters = init_paras
        self.fmo_restraint_indices = indices
        self.phi_angles = x
        self.phase_shifts = kwargs.get('phi0', None)  # using .get for safer access

        # Handling default values for min_max_paras
        if min_max_paras is None:
            self.min_max_fwd_model_parameters = np.zeros((len(init_paras[0][0]), 2))
            self.min_max_fwd_model_parameters[:, 0] -= 10
            self.min_max_fwd_model_parameters[:, 1] += 10
        else:
            self.min_max_fwd_model_parameters = min_max_paras

        # Handling default values for parameter_priors
        if parameter_priors is None:
            self.fmp_prior_models = np.array([
                ["uniform" for _ in range(len(init_paras[0][0]))]
                for _ in range(len(init_paras[0]))], dtype=str)
        else:
            self.fmp_prior_models = parameter_priors
            #print(self.fmp_prior_models)
            #exit()

        if parameter_prior_sigmas is None:
            self.fmp_prior_sigmas = np.ones(np.array(self.fwd_model_parameters).shape[1:])*5
        else:
            self.fmp_prior_sigmas = parameter_prior_sigmas

        if parameter_prior_mus is None:
            self.fmp_prior_mus = np.zeros(np.array(self.fwd_model_parameters).shape[1:])

        else:
            self.fmp_prior_mus = parameter_prior_mus


        #self.fmp_prior_mus = np.copy(np.array(self.fwd_model_parameters))


        after = self.__dict__
        self.fwd_model_attrs = {
            k: after[k] for k in after
            if k not in before or not np.array_equal(after[k], before[k])
        }







class Ensemble(object):
    def __init__(self, lam, xi, energies, diff_energies=None, diff2_energies=None, debug=False):
        """Container class for :attr:`Restraint` objects.

        Args:
            lam(float): lambda value to scale energies
            energies(np.ndarray): numpy array of energies for each state
        """

        self.ensemble = []
        if not isinstance(lam, float):
            raise ValueError("lambda should be a single number with type of 'float'")
        else:
            self.lam = lam

        if not isinstance(xi, float):
            raise ValueError("xi should be a single number with type of 'float'")
        else:
            self.xi = xi

        if np.array(energies).dtype != float:
            raise ValueError("Energies should be array with type of 'float'")
        else:
            self.energies = np.array(self.lam*energies) # Scale the energies

        try:
            if np.array(diff_energies).dtype != float:
                raise ValueError("diff_energies should be array with type of 'float'")
            else:
                self.diff_energies = np.array(self.lam*diff_energies) # Scale the energies
        except(Exception) as e:
            self.diff_energies = self.lam*np.zeros(energies.shape)

        try:
            if np.array(diff2_energies).dtype != float:
                raise ValueError("diff2_energies should be array with type of 'float'")
            else:
                self.diff2_energies = np.array(self.lam*diff2_energies) # Scale the energies
        except(Exception) as e:
            self.diff2_energies = self.lam*np.zeros(energies.shape)

        self.debug = debug


    def to_list(self):
        """Converts the :class:`Ensemble` class to a list.

        Returns:
            list: collection of :attr:`Restraint` objects
        """

        return self.ensemble


    def __initDerivedClassObj__(self, current_module, name, args, local_vars):
        """
        """

        # Pick the Restraint class upon file extension
        obj = getattr(current_module, "Restraint_%s"%(name))
        # Get all the arguments for the Parent Restraint Class if given
        args1 = {"%s"%key: val for key,val in local_vars
                if key in inspect.getfullargspec(obj)[0] if key != 'self'}
        args2 = {"%s"%key: val for key,val in local_vars
                if key in inspect.getfullargspec(obj.init_restraint)[0] if key != 'self'}
        for key,val in args.items():
            if key in inspect.getfullargspec(obj)[0]:
                args1[key] = val
            elif key in inspect.getfullargspec(obj.init_restraint)[0]:
                args2[key] = val
            else:
                possible_args = np.unique(np.concatenate(
                    [inspect.getfullargspec(obj)[0],
                    inspect.getfullargspec(obj.init_restraint)[0]]))
                possible_args = np.delete(possible_args, np.where(possible_args == "self"))
                possible_args = np.delete(possible_args, np.where(possible_args == "data"))
                raise TypeError(f"{key} is an invalid keyword argument for {obj}\n\n\
Please check your options... \n\
The ordering of dictionaries should be: {biceps.toolbox.list_extensions(input_data)}\n\
Keys for {obj.__name__} are any of: {possible_args}")
        if self.debug:
            print(obj)
            print(f"Required args by inspect:{inspect.getfullargspec(obj.init_restraint)[0]}")
            print(f"args1 given: {args1}")
            print(f"args2 given: {args2}")
        obj = obj(**args1) # Initializing Restraint
        obj.init_restraint(**args2)
        return obj



    def initialize_restraints(self, input_data, options, data_uncertainty="single"):
        """Initialize corresponding :attr:`Restraint` classes based on experimental
        observables from **input_data** for each conformational state.

        Print possible restraints with: ``biceps.toolbox.list_possible_restraints()``

        Print possible extensions with: ``biceps.toolbox.list_possible_extensions()``

        Args:
            input_data(list of str): a sorted collection of filenames (files\
                    contain `exp` (experimental) and `model` (theoretical) observables)
            options(list of dict): dictionary containing keys that match \
                    :attr:`Restraint` options and values are lists for each restraint.
        """

        verbose = self.debug
        self.options = options
        self.data_uncertainty = data_uncertainty
        extensions = biceps.toolbox.list_extensions(input_data)
        lam = self.lam
        xi = self.xi
        # NOTE: Complexity: O(n) for "single" and O(n^2) for "multiple"
        # double the states, doubles the runtime
        # double the number of data points, doubles the runtime
        #
        # for each structure
        for i in range(self.energies.size):
            self.ensemble.append([])
            energy = self.energies[i]
            diff_energy = self.diff_energies[i]
            diff2_energy = self.diff2_energies[i]
            # iterate over the types of data
            for k in range(len(input_data[0])):
                data = input_data[i][k]
                extension = extensions[k]
                ext = data.split(".")[-1]
                restraint, extension = ext.split("_")[0], ext.split("_")[-1]
                # Find all Child Restraint classes in the current file
                current_module = sys.modules[__name__]
                ###############################################################
                # Pick the Restraint class upon file extension
                _Restraint = getattr(current_module, "Restraint_%s"%(restraint))
                RestraintObj = _Restraint()
                # Load the data from filename to pd.DataFrame
                if "file_fmt" in options[k].keys():
                    data = RestraintObj.load_data(data, As=file_fmt)
                else:
                    data = RestraintObj.load_data(data, As="pickle")
                # Check to see if data_uncertainty was specific inside the options
                if "data_uncertainty" in options[k].keys():
                    data_uncertainty = options[k]["data_uncertainty"]
                    self.data_uncertainty = data_uncertainty
                # Loop through all the data points for data_uncertainty="multiple"
                if data_uncertainty == "single":
                    local_vars = locals().items()
                    R = self.__initDerivedClassObj__(current_module, restraint, options[k], local_vars)
                    R.diff_energy = diff_energy
                    R.diff2_energy = diff2_energy
                    for j in range(len(R.restraints)):
                        R.restraints[j]["diff model"] = 0.0
                        R.restraints[j]["diff2 model"] = 0.0
                    self.ensemble[-1].append(R)
                elif data_uncertainty == "multiple":
                    df = data
                    for d in range(len(df.values)):
                        data = df.iloc[[d]]
                        local_vars = locals().items()
                        R = self.__initDerivedClassObj__(current_module, restraint, options[k], local_vars)
                        R.diff_energy = diff_energy
                        R.diff2_energy = diff2_energy
                        for j in range(len(R.restraints)):
                            R.restraints[j]["diff model"] = 0.0
                            R.restraints[j]["diff2 model"] = 0.0
                        self.ensemble[-1].append(R)
                else:
                    raise ValueError("Argument `data_uncertainty` must be 'single' or 'multiple'")

#    def initialize_fwd_model_options(self, options):
#        """Initialize corresponding
#        Args:
#            options(list of dict): dictionary containing keys that match \
#                    :attr:`Restraint` options and values are lists for each restraint.
#        """
#
#        verbose = self.debug
#        self.fm_options = options



class Restraint:

    def __init__(self, data_likelihood="Gaussian", ref="uniform",
                 sigma=(0.05, 20.0, 1.02), sigma_index=None,
                 beta=(1, 2, 1), beta_index=None,
                 phi=(1, 2, 1), phi_index=None,
                 gamma=(1.0, 2.0, np.e), gamma_index=None,
                 alpha=(1.0, 2.0, np.e), alpha_index=None,
                 delta=(0, 1, 1), delta_index=None,
                 omega=(1, 2, 1), omega_index=None,
                 use_global_ref_sigma=True, verbose=False):
        """The parent :attr:`Restraint` class.

        Args:
            ref_pot(str): referenece potential e.g., "uniform". "exponential", "gaussian".\
                    If None, the default reference potential will be used for\
                    a given experimental observable
            sigma(tuple):  (sigma_min, sigma_max, dsigma)
            use_global_ref_sigma(bool): (defaults to True)
        """

        # Store restraint info
        self.restraints = []   # a list of data container objects for each restraint (e.g. NMR_Chemicalshift_Ca())

        # used for exponential reference potential
        self.betas = None
        self.neglog_exp_ref = None
        self.sum_neglog_exp_ref = 0.0

        # used for Gaussian reference potential
        self.ref_sigma = None
        self.ref_mean = None
        self.neglog_gaussian_ref = None
        self.sum_neglog_gaussian_ref = 0.0
        self.use_global_ref_sigma = use_global_ref_sigma
        self.sse = None

        # Storing the reference potential
        self.ref = ref

        # set sigma range
        self.dlogsigma = np.log(sigma[2])
        #self.dlogsigma = sigma[2]
        self.sigma_min = sigma[0]
        self.sigma_max = sigma[1]
        self.allowed_sigma = np.exp(np.arange(np.log(self.sigma_min), np.log(self.sigma_max), self.dlogsigma))
        #self.allowed_sigma = np.linspace(self.sigma_min, self.sigma_max, num=int(self.dlogsigma))
        if sigma_index == None: self.sigma_index = int(len(self.allowed_sigma)/2)
        else: self.sigma_index = sigma_index
        self.sigma = self.allowed_sigma[self.sigma_index]


        self.dbeta = beta[2]
        self.beta_min = beta[0]
        self.beta_max = beta[1]
        self.allowed_beta = np.linspace(self.beta_min, self.beta_max, int(self.dbeta))
        #self.allowed_beta = np.exp(np.arange(np.log(self.beta_min), np.log(self.beta_max), self.dlogbeta))
        if beta_index == None: self.beta_index = 0
        else: self.beta_index = beta_index
        self.beta = self.allowed_beta[self.beta_index]

        self.dphi = phi[2]
        self.phi_min = phi[0]
        self.phi_max = phi[1]
        self.allowed_phi = np.linspace(self.phi_min, self.phi_max, int(self.dphi))
        if phi_index == None: self.phi_index = 0
        else: self.phi_index = phi_index
        self.phi = self.allowed_phi[self.phi_index]

        self.dloggamma = np.log(gamma[2])
        self.gamma_min = gamma[0]
        self.gamma_max = gamma[1]
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
        #print(self.allowed_gamma)
        if gamma_index == None:
            self.gamma_index = int(len(self.allowed_gamma)/2)
        else:
            self.gamma_index = gamma_index
        self.gamma = self.allowed_gamma[self.gamma_index]
        self.scale_f_exp = self.gamma

        #self.dalpha = alpha[2]
        self.dlogalpha = np.log(alpha[2])
        self.alpha_min = alpha[0]
        self.alpha_max = alpha[1]
        #self.allowed_alpha = np.linspace(self.alpha_min, self.alpha_max, int(self.dalpha))
        self.allowed_alpha = np.exp(np.arange(np.log(self.alpha_min), np.log(self.alpha_max), self.dlogalpha))
        if alpha_index == None: self.alpha_index = int(len(self.allowed_alpha)/2)
        else: self.alpha_index = alpha_index
        self.alpha = self.allowed_alpha[self.alpha_index]
        self.beta = self.allowed_beta[self.beta_index]
        self.scale_f = self.alpha

        self.ddelta = delta[2]
        #self.dlogdelta = np.log(delta[2])
        self.delta_min = delta[0]
        self.delta_max = delta[1]

        self.allowed_delta = np.linspace(self.delta_min, self.delta_max, int(self.ddelta))
        #self.allowed_delta = np.exp(np.arange(np.log(self.delta_min), np.log(self.delta_max), self.dlogdelta))

        if delta_index == None: self.delta_index = int(len(self.allowed_delta)/2)
        else: self.delta_index = delta_index
        self.delta = self.allowed_delta[self.delta_index]
        self.offset = self.delta

        self.domega = omega[2]
        self.omega_min = omega[0]
        self.omega_max = omega[1]
        self.allowed_omega = np.linspace(self.omega_min, self.omega_max, int(self.domega))
        if omega_index == None: self.omega_index = int(len(self.allowed_omega)/2)
        else: self.omega_index = omega_index

        # delta parameter is the y-shift parameter
        # gamma and alpha parameters scale the forward model
        # i think gamma should scale the exp, but maybe it doesn't matter
        # omega is a scaling parameter for the sigmaSEM (probably don't need it)
        self.para_order = ["sigma","beta","phi","gamma","alpha","delta","omega"]
        self.ind_order = [f"{val}_index" for val in self.para_order]
        self.allow_order = [f"allowed_{val}" for val in self.para_order]
        self.verbose = verbose


    def load_data(self, filename, As="pickle"):
        """Load in the experimental restraints from many possible file formats.
        `More information about file formats \
                <https://pandas.pydata.org/pandas-docs/stable/reference/io.html>`_

        Args:
            filename(str): name of file to be loaded in memory
            As(str): file type from `pandas IO \
                    <https://pandas.pydata.org/pandas-docs/stable/reference/io.html>`_
        Returns:
            pd.DataFrame
        """

        if self.verbose:
            print('Loading %s as %s...'%(filename,As))
        df = getattr(pd, "read_%s"%As)
        #TODO: Check all formats (e.g., from="csv")
        return df(filename)


    def add_restraint(self, restraint):
        """Append an experimental restraint object to the list.

        Args:
            restraint(object): :attr:`Restraint` object
        """

        self.restraints.append(restraint)

    def compute_neglog_exp_ref(self):
        """Uses the stored beta information to compute \
                :math:`-log P_{ref}(r_{j}(X_{i}))` over each structure\
                :math:`X_{i}` and for each observable :math:`r_{j}`.
        """

        self.neglog_exp_ref = np.zeros(self.n)
        self.sum_neglog_exp_ref = 0.0
        for j in range(self.n):
            self.neglog_exp_ref[j] = np.log(self.betas[j])\
                    + self.restraints[j]['model']/self.betas[j]
            self.sum_neglog_exp_ref  += self.restraints[j]['weight'] * self.neglog_exp_ref[j]

    def compute_neglog_gaussian_ref(self):
        """An alternative option for reference potential based on
        Gaussian distribution. (Ignoring constant terms)"""

        self.neglog_gaussian_ref = np.zeros(self.n)
        self.sum_neglog_gaussian_ref = 0.0
        for j in range(self.n):
            self.neglog_gaussian_ref[j] = np.log(np.sqrt(2.0*np.pi))\
                    + np.log(self.ref_sigma[j]) + (self.restraints[j]['model'] \
                    - self.ref_mean[j])**2.0/(2.0*self.ref_sigma[j]**2.0)
            self.sum_neglog_gaussian_ref += self.restraints[j]['weight'] * self.neglog_gaussian_ref[j]


class Restraint_cs(Restraint):
    """A :attr:`Restraint` child class for chemical shifts."""

    _ext = ["H", "Ca", "N"]

    def __repr__(self):
        if self.extension is not None:
            return "<%s.Restraint_cs_%s>"%(str(__name__),str(self.extension))
        else:
            pass

    def init_restraint(self, data, energy, extension="H", weight=1,
            data_likelihood="Gaussian", data_uncertainty="single",
            stat_model="Gaussian", verbose=False):
        """Initialize the chemical shift restraints for each **exp** (experimental)
        and **model** (theoretical) observable given **data**.

        Args:
            data(pd.DataFrame): pandas DataFrame of experimental and model data
            energy(float): The (reduced) free energy of the conformation
            extensions(str): "H", "Ca", "N"
            weight(float): weight for restraint
        """

        self.extension = extension
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None
        self.data_uncertainty = data_uncertainty
        self.stat_model = stat_model
        # Reading the data from loading in filenames
        self.n = len(data.values)
        self.data_likelihood = data_likelihood

        # Group by keys
        keys = ['atom_index1', 'exp', 'model']
        grouped_data = data[keys].to_dict(orient='list')
        for row in range(len(grouped_data[keys[0]])):
            d = {key: grouped_data[key][row] for key in grouped_data.keys()}
            self.add_restraint(d)
            # N equivalent chemical shift should only get 1/N f the weight when
            #... computing chi^2 (not likely in this case but just in case we need it in the future)
            self.restraints[-1]['weight'] = weight
        self.Ndof = 0.0
        for i in range(self.n): self.Ndof += self.restraints[i]['weight']



class Restraint_J(Restraint):
    """A :attr:`Restraint` child class for J coupling constant."""

    _ext = ['J']

    def init_restraint(self, data, energy, extension="J", weight=1.0, data_likelihood="Gaussian",
            data_uncertainty="single", stat_model="Gaussian", verbose=False):
        """Initialize the sclar coupling constant restraints for each **exp**
        (experimental) and **model** (theoretical) observable given **data**.

        Args:
            data(pd.DataFrame): pandas DataFrame of experimental and model data
            energy(float): The (reduced) free energy of the conformation
            weight(float): weight for restraint
        """


        self.extension = extension
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None
        # Reading the data from loading in filenames
        self.n = len(data.values)
        self.data_likelihood = data_likelihood
        self.data_uncertainty = data_uncertainty
        self.stat_model = stat_model
        # Group by keys
        keys = ['atom_index1', 'atom_index2', 'atom_index3', 'atom_index4',
                'exp', 'model', 'restraint_index']
        #grouped_data = data[keys].to_dict()
        grouped_data = data[keys].to_dict(orient='list')
        for row in range(len(grouped_data[keys[0]])):
            d = {key: grouped_data[key][row] for key in grouped_data.keys()}
            self.add_restraint(d)

        self.equivalency_groups = {}
        # Compile equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.restraints)):
            d = self.restraints[i]
            if d['restraint_index'] != None:
                if d['restraint_index'] not in self.equivalency_groups:
                    self.equivalency_groups[d['restraint_index']] = []
                self.equivalency_groups[d['restraint_index']].append(i)
        if verbose:
            print(f'grouped_data = {grouped_data}')
            print(f'self.restraints[0] = {self.restraints[0]}')
            print(f'self.equivalency_groups = {self.equivalency_groups}')
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.Ndof = 0.0
        for i in range(self.n): self.Ndof += self.restraints[i]['weight']

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i]['weight'] = 1.0/n


class Restraint_rdc(Restraint):
    """A :attr:`Restraint` child class for residual dipolar constant."""

    _ext = ['rdc']

    def init_restraint(self, data, energy, extension="rdc", weight=1.0, data_likelihood="Gaussian",
            data_uncertainty="single", stat_model="Gaussian", verbose=False):
        """Initialize the RDC restraints for each **exp**
        (experimental) and **model** (theoretical) observable given **data**.

        Args:
            data(pd.DataFrame): pandas DataFrame of experimental and model data
            energy(float): The (reduced) free energy of the conformation
            weight(float): weight for restraint
        """

        self.extension = extension
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None

        # Reading the data from loading in filenames
        self.n = len(data.values)
        self.data_likelihood = data_likelihood
        self.data_uncertainty = data_uncertainty
        self.stat_model = stat_model
        # Group by keys
        keys = ['atom_index1', 'atom_index2', 'exp', 'model', 'restraint_index']
        #grouped_data = data[keys].to_dict()
        grouped_data = data[keys].to_dict(orient='list')
        for row in range(len(grouped_data[keys[0]])):
            d = {key: grouped_data[key][row] for key in grouped_data.keys()}
            self.add_restraint(d)

        self.equivalency_groups = {}
        # Compile equivalency_groups from the list
        for i in range(len(self.restraints)):
            d = self.restraints[i]
            if d['restraint_index'] != None:
                if d['restraint_index'] not in self.equivalency_groups:
                    self.equivalency_groups[d['restraint_index']] = []
                self.equivalency_groups[d['restraint_index']].append(i)
        if verbose:
            print(f'grouped_data = {grouped_data}')
            print(f'self.restraints[0] = {self.restraints[0]}')
            print(f'self.equivalency_groups = {self.equivalency_groups}')
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.Ndof = 0.0
        for i in range(self.n): self.Ndof += self.restraints[i]['weight']

    def adjust_weights(self):
        """Adjust the weights of restraints based on their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i]['weight'] = 1.0/n





class Restraint_noe(Restraint):
    """A :attr:`Restraint` child class for NOE distances."""

    _ext = ['noe']

    def init_restraint(self, data, energy, extension="noe", weight=1.0,
            verbose=False, data_likelihood="Gaussian", data_uncertainty="single",
            stat_model="Gaussian"):#, convert_to_intensity=False):
        """Initialize the NOE distance restraints for each **exp** (experimental)
        and **model** (theoretical) observable given **data**.

        When using :attr:`data_likelihood` the modified sum of squared errors is used:
        :math:`\chi_{\mathrm{d}}^{2}(X)=\sum_{j} w_{j}\left(\ln \left(r_{j}(X) / \gamma^{\prime} r_{j}^{\exp }\right)\right)^{2}`

        Args:
            data(pd.DataFrame): pandas DataFrame of experimental and model data
            energy(float): The (reduced) free energy :math:`f=\\beta*F` of the conformation
            weight(float): weight for restraint
            data_likelihood(bool):
            gamma(tuple): (gamma_min, gamma_max, dgamma) in log space
            convert_to_intensity(bool): convert distance to 1/r^6
        """

        #if convert_to_intensity:
        #    data = data.copy()
        #    data["model"] = data["model"].to_numpy()**-6
        #    data["exp"] = data["exp"].to_numpy()**-6

        self.extension = extension
        self.data_uncertainty = data_uncertainty
        self.data_likelihood = data_likelihood
        self.stat_model = stat_model
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None
        # Reading the data from loading in filenames
        self.n = len(data.values)
        # Group by keys
        keys = ['atom_index1', 'atom_index2', 'exp', 'model', 'restraint_index']
        #grouped_data = data[keys].to_dict()
        grouped_data = data[keys].to_dict(orient='list')
        for row in range(len(grouped_data[keys[0]])):
            d = {key: grouped_data[key][row] for key in grouped_data.keys()}
            self.add_restraint(d)

        self.equivalency_groups = {}
        for i in range(len(self.restraints)):
            d = self.restraints[i]
            if d['restraint_index'] != None:
                if d['restraint_index'] not in self.equivalency_groups:
                    self.equivalency_groups[d['restraint_index']] = []
                self.equivalency_groups[d['restraint_index']].append(i)
        if verbose:
            print(f'grouped_data = {grouped_data}')
            #print(f'self.restraints[0] = {self.restraints[0]}')
            print(f'self.equivalency_groups = {self.equivalency_groups}')
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.Ndof = 0.0
        for i in range(self.n): self.Ndof += self.restraints[i]['weight']


    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i]['weight'] = 1.0/n



class Restraint_saxs(Restraint):
    """A :attr:`Restraint` child class for small angle x-ray scattering data (SAXS)."""

    _ext = ['saxs']

    def init_restraint(self, data, energy, extension="saxs", weight=1.0,
            verbose=False, data_likelihood="Gaussian", #beta=(1.0, 2.0, np.e),
            #verbose=False, data_likelihood="Gaussian", beta=(1.0, 2.0, 1),
            #gamma=(1.0, 2.0, np.e), epsilon=(0, 1, 1),
            stat_model="Gaussian", data_uncertainty="single"):

        """Initialize the SAXS restraints for each **exp**
        (experimental) and **model** (theoretical) observable given **data**.

        Args:
            data(pd.DataFrame): pandas DataFrame of experimental and model data
            energy(float): The (reduced) free energy of the conformation
            weight(float): weight for restraint
        """

        self.extension = extension
        self.data_uncertainty = data_uncertainty
        self.data_likelihood = data_likelihood
        self.stat_model = stat_model

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None

        # Reading the data from loading in filenames
        self.n = len(data.values)

        ########################################################################
        # TODO: Need to complete this restraint and test it with synthetic data
        # Somewhere in this restraint we need to include a sclaing
        # parameter/argument A specified by the user that specifies concentration of salt.
        ########################################################################

        # Group by keys
        keys = ['exp', 'model', 'restraint_index']
        # adjust the weights of distances and dihedrals to account for equivalencies
        grouped_data = data[keys].to_dict(orient='list')
        for row in range(len(grouped_data[keys[0]])):
            d = {key: grouped_data[key][row] for key in grouped_data.keys()}
            self.add_restraint(d)
            # N equivalent chemical shift should only get 1/N f the weight when
            #... computing chi^2 (not likely in this case but just in case we need it in the future)
            self.restraints[-1]['weight'] = weight
        self.Ndof = 0.0
        for i in range(self.n): self.Ndof += self.restraints[i]['weight']



class Restraint_pf(Restraint):
    """A :attr:`Restraint` child class for protection factor."""

    _ext = ['pf']

    def init_restraint(self, data, energy, precomputed=False, pf_prior=None,
            Ncs_fi=None, Nhs_fi=None, beta_c=(0.05, 0.25, 0.01), beta_h=(0.0, 5.2, 0.2),
            beta_0=(-10.0, 0.0, 0.2), xcs=(5.0, 8.5, 0.5), xhs=(2.0, 2.7, 0.1),
            bs=(15.0, 16.0, 1.0), extension="pf", weight=1, states=None, data_uncertainty="single",
            stat_model="Gaussian",verbose=False):
        """Initialize protection factor restraints for each **exp** (experimental)
        and **model** (theoretical) observable given **data**.

        Args:
            data(pd.DataFrame): pandas DataFrame of experimental and model data
            energy(float): The (reduced) free energy :math:`f=\\beta*F` of the conformation
            weight(float): weight for restraint
            beta_c(list): [min, max, spacing]
            beta_h(list): [min, max, spacing]
            beta_0(list): [min, max, spacing]
            xcs(list): [min, max, spacing]
            xhs(list): [min, max, spacing]
            bs(list): [min, max, spacing]
        """


        self.extension = extension
        self.data_uncertainty = data_uncertainty
        self.stat_model = stat_model
        # TODO: make more general... (there exists two sets of the same variables, see line 585)
        beta_c_min, beta_c_max, dbeta_c = beta_c[0], beta_c[1], beta_c[2]
        beta_h_min, beta_h_max, dbeta_h = beta_h[0], beta_h[1], beta_h[2]
        beta_0_min, beta_0_max, dbeta_0 = beta_0[0], beta_0[1], beta_0[2]
        xcs_min, xcs_max, dxcs = xcs[0], xcs[1], xcs[2]
        xhs_min, xhs_max, dxhs = xhs[0], xhs[1], xhs[2]
        bs_min, bs_max, dbs = bs[0], bs[1], bs[2]
        allowed_xcs=np.arange(xcs_min,xcs_max,dxcs)
        allowed_xhs=np.arange(xhs_min,xhs_max,dxhs)
        allowed_bs=np.arange(bs_min,bs_max,dbs)
        Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
        Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
        for i in range(len(states)):
            for o in range(len(allowed_xcs)):
                for q in range(len(allowed_bs)):
                    infile_Nc='%s/Nc_x%0.1f_b%d_state%03d.npy'%(Ncs_fi,
                            allowed_xcs[o], allowed_bs[q],states[i])
                    Ncs[o,q,:] = (np.load(infile_Nc))
            for p in range(len(allowed_xhs)):
                for q in range(len(allowed_bs)):
                    infile_Nh='%s/Nh_x%0.1f_b%d_state%03d.npy'%(Nhs_fi,
                            allowed_xhs[p], allowed_bs[q],states[i])
                    Nhs[p,q,:] = (np.load(infile_Nh))

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None
        self.precomputed = precomputed
        # load pf priors from training model
        if pf_prior is not None:
            self.pf_prior = np.load(pf_prior)

        if not self.precomputed:
            self.Ncs = Ncs
            self.Nhs = Nhs
            self.dbeta_c = dbeta_c
            self.beta_c_min = beta_c_min
            self.beta_c_max = beta_c_max
            self.allowed_beta_c = np.arange(self.beta_c_min, self.beta_c_max, self.dbeta_c)

            self.dbeta_h = dbeta_h
            self.beta_h_min = beta_h_min
            self.beta_h_max = beta_h_max
            self.allowed_beta_h = np.arange(self.beta_h_min, self.beta_h_max, self.dbeta_h)

            self.dbeta_0 = dbeta_0
            self.beta_0_min = beta_0_min
            self.beta_0_max = beta_0_max
            self.allowed_beta_0 = np.arange(self.beta_0_min, self.beta_0_max, self.dbeta_0)

            self.dxcs = dxcs
            self.xcs_min = xcs_min
            self.xcs_max = xcs_max
            self.allowed_xcs = np.arange(self.xcs_min, self.xcs_max, self.dxcs)

            self.dxhs = dxhs
            self.xhs_min = xhs_min
            self.xhs_max = xhs_max
            self.allowed_xhs = np.arange(self.xhs_min, self.xhs_max, self.dxhs)

            self.dbs = dbs
            self.bs_min = bs_min
            self.bs_max = bs_max
            self.allowed_bs = np.arange(self.bs_min, self.bs_max, self.dbs)

            self.beta_c_index = int(len(self.allowed_beta_c)/2)
            self.beta_c = self.allowed_beta_c[self.beta_c_index]

            self.beta_h_index = int(len(self.allowed_beta_h)/2)
            self.beta_h = self.allowed_beta_h[self.beta_h_index]

            self.beta_0_index = int(len(self.allowed_beta_0)/2)
            self.beta_0 = self.allowed_beta_0[self.beta_0_index]

            self.xcs_index = int(len(self.allowed_xcs)/2)
            self.xcs = self.allowed_xcs[self.xcs_index]

            self.xhs_index = int(len(self.allowed_xhs)/2)
            self.xhs = self.allowed_xhs[self.xhs_index]

            self.bs_index = int(len(self.allowed_bs)/2)
            self.bs = self.allowed_bs[self.bs_index]


        # Reading the data from loading in filenames
        self.n = len(data.values)

        # Group by keys
        if self.precomputed:
            keys = ['atom_index1', 'exp', 'model']
        else:
            keys = ['atom_index1', 'exp']

        #grouped_data = data[keys].to_dict()
        grouped_data = data[keys].to_dict(orient='list')
        if not self.precomputed:
            for row in range(len(grouped_data[keys[0]])):
                d = {key: grouped_data[key][row] for key in grouped_data.keys()}
                # NOTE: TODO: FIXME:
                fixeme = """
PosteriorSampler.get_model() is not compatable with this input of model data...
d['model'] shown below is a multidimensional array and should be a float or double

Each type of model data might need to be broken up?
Consult with Vince about this.
"""
                d['model'] = self.compute_PF_multi(self.Ncs[:,:,row], self.Nhs[:,:,row], debug=False)
                print(fixeme)
                exit()
                self.add_restraint(d)
                self.restraints[-1]['weight'] = weight
        else:
            for row in range(len(grouped_data[keys[0]])):
                d = {key: grouped_data[key][row] for key in grouped_data.keys()}
                self.add_restraint(d)
                self.restraints[-1]['weight'] = weight

        self.sse = self.compute_sse(f=self.restraints)


    def compute_sse(self, f):
        """Returns the (weighted) sum of squared errors"""

        N,sse,Ndof = 0.0, 0.0, 0.0
        if self.precomputed:
            for i in range(self.n):
                err = f[i]['exp'] - f[i]['model']
                sse += (f[i]['weight'] * err**2.0)
                N += f[i]['weight']
            self.Ndof = N
        else:
            sse = np.zeros( (len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)) )
            for i in range(self.n):
                err = f[i]['exp'] - f[i]['model']
                sse += (f[i]['weight'] * err**2.0)
                N += f[i]['weight']
            self.Ndof = N
        return sse




    def compute_PF(self, beta_c, beta_h, beta_0, Nc, Nh):
        """Calculate predicted (ln PF)

        Args:
            beta_c,beta_h,Nc,Nh(np.ndarray): shape(nres, 2)\
                    array with columns <N_c> and <N_h> for each residue
        Returns:
            np.ndarray: ``<ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues``
        """
        return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 '''


    def compute_PF_multi(self, Ncs_i, Nhs_i, debug=False):
        """Calculate predicted (ln PF)

        .. tip:: A near future application...

        Args:
            Ncs_i,Nhs_i(np.ndarray, np.ndarray): array with columns <N_c> and\
                    <N_h> for each residue

        Returns:
            np.ndarray: ``<ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues``
        """

        N_beta_c = len(self.allowed_beta_c)
        N_beta_h = len(self.allowed_beta_h)
        N_beta_0 = len(self.allowed_beta_0)
        N_xc     = len(self.allowed_xcs)
        N_xh     = len(self.allowed_xhs)
        N_b      = len(self.allowed_bs)

        nuisance_shape = (N_beta_c, N_beta_h, N_beta_0, N_xc, N_xh, N_b)

        beta_c = self.tile_multiaxis(self.allowed_beta_c, nuisance_shape, axis=0)
        beta_h = self.tile_multiaxis(self.allowed_beta_h, nuisance_shape, axis=1)
        beta_0 = self.tile_multiaxis(self.allowed_beta_0, nuisance_shape, axis=2)
        Nc     = self.tile_2D_multiaxis(Ncs_i, nuisance_shape, axes=[3,5])
        Nh     = self.tile_2D_multiaxis(Nhs_i, nuisance_shape, axes=[4,5])

        if debug:
            print('nuisance_shape', nuisance_shape)
            print('beta_c.shape', beta_c.shape)
            print('beta_h.shape', beta_h.shape)
            print('beta_0.shape', beta_0.shape)
            print('Nc.shape', Nc.shape)
            print('Nh.shape', Nh.shape)

        return beta_c * Nc + beta_h * Nh + beta_0

    def tile_multiaxis(self, p, shape, axis=None):
        """Returns a multi-dimensional array of shape (tuple), with the 1D
        vector p along the specified axis, and tiled in all other dimensions.

        .. tip:: A near future application...

        Args:
            p(np.ndarray): a 1D array to tile
            shape(tuple): a tuple describing the shape of the returned array
            axis: the specified axis for p .  NOTE: len(p) must be equal to shape[axis]
        """

        assert shape[axis] == len(p), "len(p) must be equal to shape[axis]!"
        otherdims = [shape[i] for i in range(len(shape)) if i!=axis]
        result = np.tile(p, tuple(otherdims+[1]))
        last_axis = len(result.shape)-1
        result2 = np.rollaxis(result, last_axis, axis)
        return result2

    def tile_2D_multiaxis(self, q, shape, axes=None):
        """Returns a multi-dimensional array of shape (tuple), with the 2D vector p along the specified axis
           and tiled in all other dimensions.

        .. tip:: A near future application...

        Args:
            p(np.ndarray): a 1D array to tile
            shape(tuple): a tuple describing the shape of the returned array
            axis: the specified axis for p .  NOTE: len(p) must be equal to shape[axis]
        """

        assert (shape[axes[0]],shape[axes[1]]) == q.shape, "q.shape must be equal to (shape[axes[0]],shape[axes[1]])"
        otherdims = [shape[i] for i in range(len(shape)) if i not in axes]
        result = np.tile(q, tuple(otherdims+[1,1]))
        last_axis = len(result.shape)-1
        next_last_axis = len(result.shape)-2
        result2 = np.rollaxis(result, next_last_axis, axes[0])
        return np.rollaxis( result2, last_axis, axes[1])


    def compute_neglog_exp_ref_pf(self):
        self.neglog_exp_ref= np.zeros((self.n, len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                                       len(self.allowed_xcs),    len(self.allowed_xhs),    len(self.allowed_bs)))
        self.sum_neglog_exp_ref = 0.
        for j in range(self.n): # number of residues
            self.neglog_exp_ref[j] = np.maximum(-1.0*self.restraints[j]['model'], 0.0)
            self.sum_neglog_exp_ref  += self.restraints[j]['weight'] * self.neglog_exp_ref[j]


    def compute_neglog_gaussian_ref_pf(self):
        self.neglog_gaussian_ref = np.zeros((self.n, len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                                       len(self.allowed_xcs),    len(self.allowed_xhs),    len(self.allowed_bs)))
        self.sum_neglog_gaussian_ref = 0.
        for j in range(self.n): # number of residues
            self.neglog_gaussian_ref[j] = 0.5 * np.log(2.0*np.pi) + np.log(self.ref_sigma[j]) \
                      + (self.restraints[j]['model'] - self.ref_mean[j])**2.0/(2.0*self.ref_sigma[j]**2.0)
            self.sum_neglog_gaussian_ref += self.restraints[j]['weight'] * self.neglog_gaussian_ref[j]


    def compute_neglogP(self, parameters, parameter_indices, sse):
        """Computes :math:`-logP` for protection factor during MCMC sampling.

        Args:
            parameters(list): collection of parameters for a given step of MCMC
            parameter_indices(list): collection of indices for a given step of MCMC

        :rtype: float
        """

        result = 0
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(parameters[0])
        if not self.precomputed:
            result += sse[int(parameter_indices[1])][int(parameter_indices[2])][int(parameter_indices[3])][int(parameter_indices[4])][int(parameter_indices[5])][int(parameter_indices[6])] / (2.0*parameters[0]**2.0)
            if self.pf_prior is not None:
                result += self.pf_prior[int(parameter_indices[1])][int(parameter_indices[2])][int(parameter_indices[3])][int(parameter_indices[4])][int(parameter_indices[5])][int(parameter_indices[6])]
        else:
            result += sse / (2.0*float(parameters[0])**2.0)
        result += (self.Ndof)/2.0*np.log(2.0*np.pi)  # for normalization
        # Which reference potential was used for each restraint?
        if self.ref == "exponential":
            if isinstance(self.sum_neglog_exp_ref, float):
                result -= self.sum_neglog_exp_ref
            else:
                result -= self.sum_neglog_exp_ref[int(parameter_indices[1])][int(parameter_indices[2])][int(parameter_indices[3])][int(parameter_indices[4])][int(parameter_indices[5])][int(parameter_indices[6])]
        if self.ref == "gaussian":
            if isinstance(self.sum_neglog_gaussian_ref, float):
                result -= self.sum_neglog_gaussian_ref
            else:
                result -= self.sum_neglog_gaussian_ref[int(parameter_indices[1])][int(parameter_indices[2])][int(parameter_indices[3])][int(parameter_indices[4])][int(parameter_indices[5])][int(parameter_indices[6])]
        return result


class Preparation(object):

    def __init__(self, nstates=0,  top_file=None, outdir="./"):
        """A class to prepare **input_data** for the :attr:`biceps.Ensemble.initialize_restraints` method.

        Args:
            nstates(int): number of conformational states
            top_file(str): relative path to the structure topology file
            outdir(str): relative path for output files
        """

        self.nstates = nstates
        if top_file != None:
            self.topology = md.load(top_file).topology
        else:
            self.topology = None
        self.outdir = outdir

    def to_sorted_list(self):
        """Uses ``biceps.toolbox.sort_data()`` to return sorted list of **input_data**."""

        return biceps.toolbox.sort_data(self.outdir)

    def write_DataFrame(self, filename, As="pickle", verbose=False):
        """Write Pandas DataFrame **As** user specified filetype.
        `More information about file formats \
                <https://pandas.pydata.org/pandas-docs/stable/reference/io.html>`_

        Args:
            filename(str): name of file to be loaded in memory
            As(str): file type from `pandas IO \
                    <https://pandas.pydata.org/pandas-docs/stable/reference/io.html>`_
        Returns:
            pd.DataFrame
        """

        #biceps.toolbox.mkdir(self.outdir)
        #columns = { self.keys[i] : self.header[i] for i in range(len(self.keys)) }
        if verbose: print('Writing %s as %s...'%(filename,As))
        df = pd.DataFrame(self.biceps_df)
        #dfOut = getattr(self.df.rename(columns=columns), "to_%s"%As)
        dfOut = getattr(df, "to_%s"%As)
        dfOut(filename)

    def prepare_cs(self, exp_data, model_data, indices, extension, write_as="pickle", verbose=False):
        """A method for preprocessing chemicalshift **exp_data** and **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: ppm)
            model_data(list,str): model data, path to model data file (units: ppm)
            indices(list,str): indices, path to atom indices
            extension(str): nuclei for the CS data ("H" or "Ca or "N")
        """

        #self.header = ('exp', 'exp err', 'model', 'restraint_index', 'atom_index1', 'res1',
        self.header = ('exp', 'model', 'restraint_index', 'atom_index1', 'res1',
                'atom_name1', )
        self.ind = biceps.toolbox.check_indices(indices)
        self.exp_data = biceps.toolbox.check_exp_data(exp_data)
        self.model_data = biceps.toolbox.check_model_data(model_data)

        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            if type(self.model_data[j]) == str: model_data = np.loadtxt(self.model_data[j])
            else: model_data = self.model_data[j]
            if model_data.size == 1: model_data = np.array([model_data])
            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i])
                dd['atom_index1'].append(a1)
                if self.topology != None:
                    dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                else:
                    dd.pop('res1', None)
                    dd.pop('atom_name1', None)
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                #if exp_err.shape[1] > 1: dd['exp err'].append(np.float64(self.exp_data[i,2]))
                dd['model'].append(np.float64(model_data[i]))
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.cs_%s"%(j, extension)
            if self.outdir:
                self.write_DataFrame(filename=os.path.join(self.outdir,filename), As=write_as, verbose=verbose)



    def prepare_noe(self, exp_data, model_data, indices, extension="noe", write_as="pickle", verbose=False):
        """A method for preprocessing NOE **exp_data** and **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: :math:`Å`)
            model_data(list,str): model data, path to model data file (units: :math:`Å`)
            indices(list,str): indices, path to atom indices
            extension(str): file extension for the NOE data ("noe")
        """

        self.header = ('exp', 'model', 'restraint_index', 'atom_index1', 'res1',
                'atom_name1', 'atom_index2', 'res2', 'atom_name2', )
        self.ind = biceps.toolbox.check_indices(indices)
        self.exp_data = biceps.toolbox.check_exp_data(exp_data)
        self.model_data = biceps.toolbox.check_model_data(model_data)

        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            if type(self.model_data[j]) == str: model_data = np.loadtxt(self.model_data[j])
            else: model_data = self.model_data[j]

            for i in range(self.ind.shape[0]):
                a1, a2 = int(self.ind[i,0]), int(self.ind[i,1])
                dd['atom_index1'].append(a1)
                dd['atom_index2'].append(a2)
                if self.topology != None:
                    dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                    dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                else:
                    dd.pop('res1', None)
                    dd.pop('atom_name1', None)
                    dd.pop('res2', None)
                    dd.pop('atom_name2', None)
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                #if exp_err.shape[1] > 1: dd['exp err'].append(np.float64(self.exp_data[i,2]))
                dd['model'].append(np.float64(model_data[i]))
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.noe"%(j)
            if self.outdir:
                self.write_DataFrame(filename=os.path.join(self.outdir,filename), As=write_as, verbose=verbose)


    def prepare_saxs(self, exp_data, model_data, extension="saxs",
                    write_as="pickle", verbose=False):
        """A method for preprocessing SAXS **exp_data** and **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: :math:`AU`)
            model_data(list,str): model data, path to model data file (units: :math:`AU`)
            extension(str): nuclei for the saxs data
        """

        self.header = ['exp', 'model', 'restraint_index']
        self.exp_data = biceps.toolbox.check_exp_data(exp_data)
        self.model_data = biceps.toolbox.check_model_data(model_data)
        self.q = []
        if len(self.exp_data.shape) > 1:
            self.q = self.exp_data[:,0]
            self.header.append('q (1/Å)')
            self.exp_data = self.exp_data[:,1]

        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            if type(self.model_data[j]) == str: model_data = np.loadtxt(self.model_data[j])
            else: model_data = self.model_data[j]

            # See if it's an intensity profile. If this doesn't work, it's Rg, a scalar value
            try: len(self.exp_data)
            except(Exception) as e: self.exp_data = [self.exp_data]

            try: len(model_data)
            except(Exception) as e: model_data = [model_data]

            for i in range(len(self.exp_data)):
                dd['restraint_index'].append(i)
                dd['exp'].append(np.float64(self.exp_data[i]))
                dd['model'].append(np.float64(model_data[i]))
                if self.q != []: dd['q (1/Å)'].append(self.q[i])

            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.saxs"%(j)
            if self.outdir:
                self.write_DataFrame(filename=os.path.join(self.outdir,filename), As=write_as, verbose=verbose)




    def prepare_J(self, exp_data, model_data, indices, extension="J", write_as="pickle", verbose=False):
        """A method for preprocessing scalar coupling **exp_data** and **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: Hz)
            model_data(list,str): model data, path to model data file (units: Hz)
            indices(list,str): indices, path to atom indices
            extension(str): nuclei for the CS data ("H" or "Ca or "N")
        """

        self.header = ('exp', 'model', 'restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'atom_index2', 'res2', 'atom_name2', 'atom_index3', 'res3', 'atom_name3',
                'atom_index4', 'res4', 'atom_name4',  )
        self.ind = biceps.toolbox.check_indices(indices)
        self.exp_data = biceps.toolbox.check_exp_data(exp_data)
        self.model_data = biceps.toolbox.check_model_data(model_data)

        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            if type(self.model_data[j]) == str: model_data = np.loadtxt(self.model_data[j])
            else: model_data = self.model_data[j]

            for i in range(self.ind.shape[0]):
                a1, a2, a3, a4   = int(self.ind[i,0]), int(self.ind[i,1]), int(self.ind[i,2]), int(self.ind[i,3])
                dd['atom_index1'].append(a1);dd['atom_index2'].append(a2)
                dd['atom_index3'].append(a3);dd['atom_index4'].append(a4)
                if self.topology != None:
                    dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                    dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                    dd['res3'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a3][0]))
                    dd['atom_name3'].append(str([atom.name for atom in self.topology.atoms if atom.index == a3][0]))
                    dd['res4'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a4][0]))
                    dd['atom_name4'].append(str([atom.name for atom in self.topology.atoms if atom.index == a4][0]))
                else:
                    dd.pop('res1', None)
                    dd.pop('atom_name1', None)
                    dd.pop('res2', None)
                    dd.pop('atom_name2', None)
                    dd.pop('res3', None)
                    dd.pop('atom_name3', None)
                    dd.pop('res4', None)
                    dd.pop('atom_name4', None)
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                #if exp_err.shape[1] > 1: dd['exp err'].append(np.float64(self.exp_data[i,2]))
                dd['model'].append(np.float64(model_data[i]))
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.%s"%(j, extension)
            if self.outdir:
                self.write_DataFrame(filename=os.path.join(self.outdir,filename), As=write_as, verbose=verbose)


    def prepare_rdc(self, exp_data, model_data, indices, extension="rdc",
                    write_as="pickle", verbose=False):
        """A method for preprocessing RDC **exp_data** and **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: :math:`Hz`)
            model_data(list,str): model data, path to model data file (units: :math:`Hz`)
            indices(list,str): indices, path to atom indices
            extension(str): nuclei for the rdc data
        """

        self.header = ('exp', 'model', 'restraint_index', 'atom_index1', 'res1',
                'atom_name1', 'atom_index2', 'res2', 'atom_name2', )
        self.ind = biceps.toolbox.check_indices(indices)
        self.exp_data = biceps.toolbox.check_exp_data(exp_data)
        self.model_data = biceps.toolbox.check_model_data(model_data)

        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            if type(self.model_data[j]) == str: model_data = np.loadtxt(self.model_data[j])
            else: model_data = self.model_data[j]

            for i in range(self.ind.shape[0]):
                a1, a2 = int(self.ind[i,0]), int(self.ind[i,1])
                dd['atom_index1'].append(a1)
                dd['atom_index2'].append(a2)
                if self.topology != None:
                    dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                    dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                else:
                    dd.pop('res1', None)
                    dd.pop('atom_name1', None)
                    dd.pop('res2', None)
                    dd.pop('atom_name2', None)
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                #if exp_err.shape[1] > 1: dd['exp err'].append(np.float64(self.exp_data[i,2]))
                dd['model'].append(np.float64(model_data[i]))
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.%s"%(j, extension)
            if self.outdir:
                self.write_DataFrame(filename=os.path.join(self.outdir,filename), As=write_as, verbose=verbose)





    def prepare_pf(self, exp_data, model_data=None, indices=None, extension=None, write_as="pickle", verbose=False):
        """A method for preprocessing HDX protection factor **exp_data** and
        **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: Hz)
            model_data(list,str): model data, path to model data file (units: Hz)
            indices(list,str): indices, path to atom indices
            extension(str):
        """

        if model_data: self.header = ('exp','model', 'restraint_index', 'atom_index1', 'res1', )
        else: self.header = ('exp', 'restraint_index', 'atom_index1', 'res1',)
        self.ind = biceps.toolbox.check_indices(indices)
        self.exp_data = biceps.toolbox.check_exp_data(exp_data)
        self.model_data = biceps.toolbox.check_model_data(model_data)

        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the\
                    number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            if type(self.model_data[j]) == str: model_data = np.loadtxt(self.model_data[j])
            else: model_data = self.model_data[j]

            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i,0])
                dd['atom_index1'].append(a1)
                if self.topology != None:
                    dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                    dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                else:
                    dd.pop('res1', None)
                    dd.pop('atom_name1', None)
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                #if exp_err.shape[1] > 1: dd['exp err'].append(np.float64(self.exp_data[i,2]))
                if model_data:
                    dd['model'].append(np.float64(model_data[i]))
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.pf"%(j)
            if self.outdir:
                self.write_DataFrame(filename=os.path.join(self.outdir,filename), As=write_as, verbose=verbose)




