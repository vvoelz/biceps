# -*- coding: utf-8 -*-
import os, sys, inspect
import numpy as np
import pandas as pd
import mdtraj as md
import biceps
from biceps.KarplusRelation import * # Returns J-coupling values from dihedral angles


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
    #restraints = [ key for key in vars(current_module).keys() if key.startswith("Restraint_") ]
    restraints = biceps.toolbox.list_possible_restraints()

    # NOTE: if input_data is given, then remove information regarding all other restraints
    if input_data != None:
        scheme = biceps.toolbox.list_res(input_data)
        extensions = [ res.split("_")[-1] for res in scheme ]
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





class Ensemble(object):
    def __init__(self, lam, energies, debug=False):
        """Container class for :attr:`biceps.Restraint.Restraint` objects.

        Args:
            lam(float): lambda value to scale energies
            energies(np.ndarray): numpy array of energies for each state
        """

        self.ensemble = []
        if not isinstance(lam, float):
            raise ValueError("lambda should be a single number with type of 'float'")
        else:
            self.lam = lam

        if np.array(energies).dtype != float:
            raise ValueError("Energies should be array with type of 'float'")
        else:
            self.energies = self.lam*energies # Scale the energies
        self.debug = debug


    def to_list(self):
        """Converts the :class:`Ensemble` class to a list.

        Returns:
            list: collection of :attr:`biceps.Restraint.Restraint` objects
        """

        return self.ensemble


    def initialize_restraints(self, input_data, options=None):
        """Initialize corresponding :attr:`biceps.Restraint.Restraint` classes based on experimental
        observables from **input_data** for each conformational state.

        Args:
            input_data(list of str): a sorted collection of filenames (files\
                    contain `exp` (experimental) and `model` (theoretical) observables)
            options(list of dict): dictionary containing keys that match \
                    :attr:`biceps.Restraint.Restraint` parameters and values are lists for each restraint.\

        .. code-block:: python

            # In general:
            parameters = [dict(**kwargs),...,dict(**kwargs)]
            # More specifically, for J and NOE data restraints, respectively:
            parameters = [dict(ref='uniform', sigma=(0.05, 20.0, 1.02)),
                          dict(ref='exp', sigma=(0.05, 5.0, 1.02), gamma=(0.2, 5.0, 1.02))]

        .. tip::

            See the following parent :attr:`biceps.Restraint.Restraint` and
            child class methods for the full list of keyword arguments (**kwargs)
            for each restraint used inside `parameters`:

            :attr:`biceps.Restraint.Restraint_cs.init_restraint`

            :attr:`biceps.Restraint.Restraint_J.init_restraint`

            :attr:`biceps.Restraint.Restraint_noe.init_restraint`

            :attr:`biceps.Restraint.Restraint_pf.init_restraint`

            Print possible restraints with: :attr:`biceps.toolbox.list_possible_restraints`

            Print possible extensions with: :attr:`biceps.toolbox.list_possible_extensions`

        """

        verbose = self.debug
        self.options = options
        if options is None: options = [dict() for i in range(len(input_data[0]))]
        extensions = biceps.toolbox.list_extensions(input_data)
        lam = self.lam
        for i in range(self.energies.shape[0]):
            self.ensemble.append([])
            energy = self.energies[i]
            for k in range(len(input_data[0])):
                data = input_data[i][k]
                extension = extensions[k]
                ext = data.split(".")[-1]
                restraint, extension = ext.split("_")[0], ext.split("_")[-1]
                # Find all Child Restraint classes in the current file
                current_module = sys.modules[__name__]
                # Pick the Restraint class upon file extension
                R = getattr(current_module, "Restraint_%s"%(restraint))
                # Get all the arguments for the Parent Restraint Class if given
                args1 = {"%s"%key: val for key,val in locals().items()
                        if key in inspect.getfullargspec(R)[0] if key != 'self'}
                args2 = {"%s"%key: val for key,val in locals().items()
                        if key in inspect.getfullargspec(R.init_restraint)[0] if key != 'self'}
                # It shouldn't matter the ordering of the keys and values
                # All options are parsed as respective Restraint child class arguments
                for key,val in options[k].items():
                    if key in inspect.getfullargspec(R)[0]:
                        args1[key] = val
                    elif key in inspect.getfullargspec(R.init_restraint)[0]:
                        args2[key] = val
                    else:
                        possible_args = np.unique(np.concatenate(
                            [inspect.getfullargspec(R)[0],
                            inspect.getfullargspec(R.init_restraint)[0]]))
                        possible_args = np.delete(possible_args, np.where(possible_args == "self"))
                        possible_args = np.delete(possible_args, np.where(possible_args == "data"))
                        raise TypeError(f"{key} is an invalid keyword argument for {R}\n\n\
Please check your options... \n\
The input data provided suggests the ordering of dictionaries should be: {biceps.toolbox.list_extensions(input_data)}\n\
Dictionary keys for {R.__name__} are any of: {possible_args}")
                if self.debug:
                    print(R)
                    print(f"Required args by inspect:{inspect.getfullargspec(R.init_restraint)[0]}")
                    print(f"args1 given: {args1}")
                    print(f"args2 given: {args2}")
                R = R(**args1) # Initializing Restraint
                R.init_restraint(**args2)
                self.ensemble[-1].append(R)



class Restraint(object):

    def __init__(self, ref="uniform", sigma=[0.05, 20.0, 1.02],
            use_global_ref_sigma=True, verbose=False):
        """The parent :attr:`biceps.Restraint.Restraint` class.

        Args:
            ref_pot(str): referenece potential e.g., "uniform". "exp", "gau".\
                    If None, the default reference potential will be used for\
                    a given experimental observable
            sigma(list):  (sigma_min, sigma_max, dsigma)
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
        self.sigma_min = sigma[0]
        self.sigma_max = sigma[1]
        self.allowed_sigma = np.exp(np.arange(np.log(self.sigma_min),
            np.log(self.sigma_max), self.dlogsigma))
        self.sigma_index = int(len(self.allowed_sigma)/2)
        self.sigma = self.allowed_sigma[self.sigma_index]

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
            restraint(object): :attr:`biceps.Restraint.Restraint` object
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
    """A :attr:`biceps.Restraint.Restraint` child class for chemical shifts."""

    _ext = ["H", "Ca", "N"]

    def __repr__(self):
        if self.extension is not None:
            return "<%s.Restraint_cs_%s>"%(str(__name__),str(self.extension))
        else:
            pass

    def init_restraint(self, data, energy, extension="H", weight=1, file_fmt="pickle", verbose=False):
        """Initialize the chemical shift restraints for each experimental
        and theoretical observable given data.

        Args:
            data(str): filename of data
            energy(float): The (reduced) free energy of the conformation
            extensions(str): "H", "Ca", "N"
            weight(float): weight for restraint
        """

        self.extension = extension

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None

        # Reading the data from loading in filenames
        data = self.load_data(data, As=file_fmt)
        self.n = len(data.values)

        # Group by keys
        keys = ['atom_index1', 'exp', 'model']
        grouped_data = data[keys].to_dict()
        for row in range(len(grouped_data[keys[0]])):
            d = {key: grouped_data[key][row] for key in grouped_data.keys()}
            self.add_restraint(d)
            # N equivalent chemical shift should only get 1/N f the weight when
            #... computing chi^2 (not likely in this case but just in case we need it in the future)
            self.restraints[-1]['weight'] = weight  #1.0/3.0 used in JCTC 2020 paper  # default is N=1
        self.sse = self.compute_sse(f=self.restraints)

    def compute_sse(self, f):
        """Returns the (weighted) sum of squared errors for chemical shift values."""

        N,sse = 0.0, 0.0
        for i in range(self.n):
            err = f[i]['model'] - f[i]['exp']
            sse += (f[i]['weight']*err**2.0)
            N += f[i]['weight']
        self.Ndof = N
        return sse


    def compute_neglogP(self, parameters, parameter_indices, sse):
        """Computes :math:`-logP` for chemical shift during MCMC sampling.

        :math:`-ln P(X, \\sigma | D)=(N_{j}+1) \ln \sigma+\\chi^{2}(X) / 2 \\sigma^{2}-ln P(X) -ln Q_{ref}+(N_{j} / 2) ln 2 \pi + f_{i}`,

        where :math:`f_i` is the free energy of conformational state :math:`i`
        obtained from computational simulations and :math:`\\chi^{2}(X)` is the sum of
        squared errors and can be computed as following in practice as \
                :math:`\\chi^{2}(X)=\sum_{j} w_{j}(r_{j}(X)-r_{j}^{\exp})^{2}`

        Args:
            parameters(list): collection of parameters for a given step of MCMC
            parameter_indices(list): collection of indices for a given step of MCMC
            sse(float): sum of squared errors

        :rtype: float
        """

        result = 0
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(parameters[0])
        result += sse / (2.0*float(parameters[0])**2.0)
        result += (self.Ndof)/2.0*np.log(2.0*np.pi)  # for normalization
        if self.ref == "exponential":
            result -= self.sum_neglog_exp_ref
        if self.ref == "gaussian":
            result -= self.sum_neglog_gaussian_ref
        return result


class Restraint_J(Restraint):
    """A :attr:`biceps.Restraint.Restraint` child class for J coupling constant."""

    _ext = ['J']

    def init_restraint(self, data, energy, extension="J", weight=1, file_fmt="pickle", verbose=False):
        """Initialize the sclar coupling constant restraints for each **exp**
        (experimental) and **model** (theoretical) observable given **data**.

        Args:
            data(str): filename of data
            energy(float): The (reduced) free energy of the conformation
            weight(float): weight for restraint
        """

        self.extension = extension
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None

        # Reading the data from loading in filenames
        data = self.load_data(data, As=file_fmt)
        self.n = len(data.values)

        # Group by keys
        keys = ['atom_index1', 'atom_index2', 'atom_index3', 'atom_index4',
                'exp', 'model', 'restraint_index']
        grouped_data = data[keys].to_dict()
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
        self.sse = self.compute_sse(f=self.restraints)


    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i]['weight'] = 1.0/n


    def compute_sse(self, f):
        """Returns the (weighted) sum of squared errors"""

        N,sse = 0.0, 0.0
        for i in range(self.n):
            err = f[i]['model'] - f[i]['exp']
            sse += (f[i]['weight']*err**2.0)
            N += f[i]['weight']
        self.Ndof = N
        return sse


    def compute_neglogP(self, parameters, parameter_indices, sse):
        """Computes :math:`-logP` for scalar coupling constant during MCMC sampling.

        Args:
            parameters(list): collection of parameters for a given step of MCMC
            parameter_indices(list): collection of indices for a given step of MCMC
            sse(float): sum of squared errors

        :rtype: float
        """

        result = 0
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(parameters[0])
        result += sse / (2.0*float(parameters[0])**2.0)
        result += (self.Ndof)/2.0*np.log(2.0*np.pi)  # for normalization
        if self.ref == "exponential":
            result -= self.sum_neglog_exp_ref
        if self.ref == "gaussian":
            result -= self.sum_neglog_gaussian_ref
        return result



class Restraint_noe(Restraint):

    _ext = ['noe']

    def init_restraint(self, data, energy, extension="noe", weight=1, file_fmt="pickle", verbose=False,
            log_normal=False, gamma=[0.2, 10.0, 1.01]):
        """
        Initialize the NOE distance restraints for each experimental
        and theoretical observable given data.
        When :attr:`log_normal=True`, the modified sum of squared errors is used
        :math:`\chi_{d}^{2}(X)=\sum_{j} w_{j}( \ln ( r_{j}(X) / \gamma' r_{j}^{exp} ))^{2}`
        Args:
            data(str): filename of data
            energy(float): The (reduced) free energy :math:`f=\\beta*F` of the conformation
            weight(float): weight for restraint
            log_normal(bool): use log normal distribution
            gamma(list): [gamma_min, gamma_max, dgamma] in log space"""


        self.extension = extension
        # Store info about gamma^(-1/6) scaling parameter array
        self.dloggamma = np.log(gamma[2])
        self.gamma_min = gamma[0]
        self.gamma_max = gamma[1]
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
        self.gamma_index = int(len(self.allowed_gamma)/2)
        self.gamma = self.allowed_gamma[self.gamma_index]

        # Flag to use log-normal distance errors log(d/d0)
        self.log_normal = log_normal

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.energy = energy
        self.Ndof = None

        # Reading the data from loading in filenames
        data = self.load_data(data, As=file_fmt)
        self.n = len(data.values)

        # Group by keys
        keys = ['exp', 'model', 'restraint_index']
        grouped_data = data[keys].to_dict()
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
        self.sse = self.compute_sse(f=self.restraints)

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i]['weight'] = 1.0/n


    def compute_sse(self, f):
        """Returns the (weighted) sum of squared errors"""

        _sse = np.array([0.0 for gamma in self.allowed_gamma])
        for g in range(len(self.allowed_gamma)):
            N,sse = 0.0, 0.0
            for i in range(self.n):
                gamma = self.allowed_gamma[g]
                #if self.use_log_normal_noe:
                if self.log_normal:
                    err = np.log(f[i]['model']/(gamma*f[i]['exp']))
                else:
                    err = gamma*f[i]['exp'] - f[i]['model']
                sse += (f[i]['weight'] * err**2.0)
                N += f[i]['weight']
            _sse[g] = sse
            self.Ndof = N
        return _sse


    def compute_neglogP(self, parameters, parameter_indices, sse):
        """Computes :math:`-logP` for NOE during MCMC sampling.

        Args:
            parameters(list): collection of parameters for a given step of MCMC
            parameter_indices(list): collection of indices for a given step of MCMC
            sse(float): sum of squared errors

        :rtype: float
        """

        result = 0
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(parameters[0])
        result += sse[int(parameter_indices[1])] / (2.0*parameters[0]**2.0)
        result += (self.Ndof)/2.0*np.log(2.0*np.pi)  # for normalization
        if self.ref == "exponential":
            result -= self.sum_neglog_exp_ref
        if self.ref == "gaussian":
            result -= self.sum_neglog_gaussian_ref
        return result


class Restraint_pf(Restraint):
    """A :attr:`biceps.Restraint.Restraint` child class for protection factor."""

    _ext = ['pf']

    def init_restraint(self, data, energy, precomputed=False, pf_prior=None,
            Ncs_fi=None, Nhs_fi=None, beta_c=(0.05, 0.25, 0.01), beta_h=(0.0, 5.2, 0.2),
            beta_0=(-10.0, 0.0, 0.2), xcs=(5.0, 8.5, 0.5), xhs=(2.0, 2.7, 0.1),
            bs=(15.0, 16.0, 1.0), extension="pf", weight=1, file_fmt="pickle", states=None, verbose=False):
        """Initialize protection factor restraints for each **exp** (experimental)
        and **model** (theoretical) observable given **data**.

        Args:
            data(str): filename of data
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
        data = self.load_data(data, As=file_fmt)
        self.n = len(data.values)

        # Group by keys
        if self.precomputed:
            keys = ['atom_index1', 'exp', 'model']
        else:
            keys = ['atom_index1', 'exp']

        grouped_data = data[keys].to_dict()
        if not self.precomputed:
            for row in range(len(grouped_data[keys[0]])):
                d = {key: grouped_data[key][row] for key in grouped_data.keys()}
                d['model'] = self.compute_PF_multi(self.Ncs[:,:,row], self.Nhs[:,:,row], debug=False)
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

        N,sse = 0.0, 0.0
        if self.precomputed:
            for i in range(self.n):
                err = f[i]['model'] - f[i]['exp']
                sse += (f[i]['weight']*err**2.0)
                N += f[i]['weight']
            self.Ndof = N
        else:
            sse = np.zeros( (len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)) )
            self.Ndof = 0.
            for i in range(self.n):
                err = f[i]['model'] - f[i]['exp']
                sse += (f[i]['weight'] * err**2.0)
                self.Ndof += f[i]['weight']
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
            sse(float): sum of squared errors

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

    def __init__(self, nstates=0,  top_file=None, outdir=None):
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



    def prepare_noe(self, exp_data, model_data, indices, extension=None, write_as="pickle", verbose=False):
        """A method for preprocessing NOE **exp_data** and **model_data**.

        Args:
            exp_data(list,str): experimental data, path to experimental data file (units: :math:`Å`)
            model_data(list,str): model data, path to model data file (units: :math:`Å`)
            indices(list,str): indices, path to atom indices
            extension(str): nuclei for the CS data ("H" or "Ca or "N")
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
                self.write_DataFrame(os.path.join(self.outdir,filename), As=write_as, verbose=verbose)


    def prepare_J(self, exp_data, model_data, indices, extension=None, write_as="pickle", verbose=False):
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
            filename = "%s.J"%(j)
            if self.outdir:
                self.write_DataFrame(os.path.join(self.outdir,filename), As=write_as, verbose=verbose)



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

        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
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
                self.write_DataFrame(os.path.join(self.outdir,filename), As=write_as, verbose=verbose)

if __name__ == "__main__":

    import doctest
    doctest.testmod()




