# -*- coding: utf-8 -*-
import sys, inspect
import numpy as np
import pandas as pd
import mdtraj as md
import biceps
from biceps.KarplusRelation import * # Returns J-coupling values from dihedral angles
from biceps.toolbox import *

class Ensemble(object):
    def __init__(self, lam, energies, top, verbose=False):
        """Container of Restraint objects"""

        self.ensemble = []
        self.top = top
        if not isinstance(lam, float):
            raise ValueError("lambda should be a single number with type of 'float'")
        else:
            self.lam = lam

        # TODO: check
        if np.array(energies).dtype != float:
            raise ValueError("Energies should be array with type of 'float'")
        else:
            self.energies = energies

        self.verbose = verbose

    def to_list(self):
        return self.ensemble

    def initialize_restraints(self, exp_data, ref_pot=None, uncern=None,
            gamma=None, precomputed=False, Ncs_fi=None, Nhs_fi=None,
            extensions=None):
        """Initialize corresponding restraint class based on experimental observables in input files for each conformational state.

        :param str PDB_filename: topology file name ('*.pdb')
        :param float lam: lambdas
        :param float energy: potential energy for each conformational state
        :param str default=None ref: reference potential (if default, will use our suggested reference potential for each experimental observables)
        :param str data: BICePs input files directory
        :param list default=None uncern: nuisance parameters range (if default, will use our suggested broad range (may increase sampling requirement for convergence))
        :param list default=None gamma: only for NOE, range of gamma (if default, will use our suggested broad range (may increase sampling requirement for convergence))"""

        verbose = self.verbose
        uncertainties = uncern
        lam = self.lam
        for i in range(self.energies.shape[0]):
            self.ensemble.append([])
            for k in range(len(exp_data[0])):
                data = exp_data[i][k]
                energy = self.energies[i]
                uncern = uncertainties[k]
                extension = extensions[k]

                if ref_pot[k] is not None and not isinstance(ref_pot[k], str):
                    raise ValueError("Reference potential type must be a 'str'")
                if uncern ==  None:
                    sigma_min, sigma_max, dlogsigma=0.05, 20.0, np.log(1.02)
                else:
                    if len(uncern) != 3:
                        raise ValueError("uncertainty should be a list of three items: sigma_min, sigma_max, dlogsigma")
                    else:
                        sigma_min, sigma_max, dlogsigma = uncern[0], uncern[1], np.log(uncern[2])
                if gamma ==  None:
                    gamma_min, gamma_max, dloggamma = 0.05, 20.0, np.log(1.02)
                else:
                    if len(gamma) != 3:
                        raise ValueError("gamma should be a list of three items: gamma_min, gamma_max, dgamma")
                    else:
                        gamma_min, gamma_max, dloggamma = gamma[0], gamma[1], np.log(gamma[2])

                # TODO: Place in Protection factor observable or restraint
                if data.endswith('pf'):
                    if not precomputed:
                        if Ncs_fi == None or Nhs_fi == None or state == None:
                            raise ValueError("Ncs and Nhs and state numebr are needed!")
                    # add uncern option here later
                    # don't trust these numbers, need to be confirmed!!! Yunhui 06/2019
                    beta_c_min, beta_c_max, dbeta_c = 0.05, 0.25, 0.01
                    beta_h_min, beta_h_max, dbeta_h = 0.0, 5.2, 0.2
                    beta_0_min, beta_0_max, dbeta_0 = -10.0, 0.0, 0.2
                    xcs_min, xcs_max, dxcs = 5.0, 8.5, 0.5
                    xhs_min, xhs_max, dxhs = 2.0, 2.7, 0.1
                    bs_min, bs_max, dbs = 15.0, 16.0, 1.0

                    allowed_xcs=np.arange(xcs_min,xcs_max,dxcs)
                    allowed_xhs=np.arange(xhs_min,xhs_max,dxhs)
                    allowed_bs=np.arange(bs_min,bs_max,dbs)
                    # 107=residue numbers, Nc/Nh file names are hard coded for now. Yunhui 06/19
                    Ncs=np.zeros((len(allowed_xcs),len(allowed_bs),107))
                    Nhs=np.zeros((len(allowed_xhs),len(allowed_bs),107))
                    for o in range(len(allowed_xcs)):
                        for q in range(len(allowed_bs)):
                            infile_Nc='%s/Nc_x%0.1f_b%d_state%03d.npy'%(Ncs_fi, allowed_xcs[o], allowed_bs[q],state)
                            Ncs[o,q,:] = (np.load(infile_Nc))
                    for p in range(len(allowed_xhs)):
                        for q in range(len(allowed_bs)):
                            infile_Nh='%s/Nh_x%0.1f_b%d_state%03d.npy'%(Nhs_fi, allowed_xhs[p], allowed_bs[q],state)
                            Nhs[p,q,:] = (np.load(infile_Nh))

                if ref_pot[k] ==  None:
                    # TODO: place the default ref_pot[k] inside the class
                    ref_pot[k] = 'exp' # 'uniform'

                ext = data.split(".")[-1]
                #restraint, extension = ext.split("_")
                restraint, extension = ext.split("_")[0], ext.split("_")[-1]
                # Find all Child Restraint classes in the current file
                current_module = sys.modules[__name__]
                # Pick the Restraint class upon file extension
                _Restraint = getattr(current_module, "Restraint_%s"%(restraint))
                # Initializing Restraint
                R = _Restraint(PDB_filename=self.top, ref=ref_pot[k], dlogsigma=dlogsigma,
                        sigma_min=sigma_min, sigma_max=sigma_max)
                # Get all the arguments for the Child Restraint Class
                args = {"%s"%key: val for key,val in locals().items()
                        if key in inspect.getfullargspec(R.init_restraint)[0]
                        if key != 'self'}
                #print(f"args = {args}")
                #print(f"Required args:{R.init_restraint.__code__.co_varnames}")
                #print(f"Required args by inspect:{inspect.getfullargspec(R.init_restraint)[0]}")
                #exit()
                R.init_restraint(**args)
                self.ensemble[-1].append(R)





class Restraint(object):
    """The parent class of all Restraint() objects.

    :param str PDB_filename: A topology file (*.pdb)
    :param str ref: Reference potential.
    :param float default=np.log(1.02) dlogsigma:
    :param float sigma_min: default = 0.05
    :param float sigma_max: default = 20.0
    :param bool default=True use_global_ref_sigma: """

    def __init__(self, PDB_filename, ref, dlogsigma=np.log(1.02),
            sigma_min=0.05, sigma_max=20.0, use_global_ref_sigma=True):
        """Initialize the Restraint class.

        :param str PDB_filename: A topology file (*.pdb)
        :param str ref: Reference potential.
        :param float default=np.log(1.02) dlogsigma:
        :param float default=0.05 sigma_min:
        :param float default=20.0 sigma_max:
        :param bool default=True use_global_ref_sigma:
        """

        # Store restraint info
        self.restraints = []   # a list of data container objects for each restraint (e.g. NMR_Chemicalshift_Ca())

        # Conformational Information
        self.PDB_filename = PDB_filename
        self.conf = md.load_pdb(PDB_filename)

        # Convert the coordinates from nm to Angstrom units
        self.conf.xyz = self.conf.xyz*10.0

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
        self.dlogsigma = dlogsigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.allowed_sigma = np.exp(np.arange(np.log(self.sigma_min),
            np.log(self.sigma_max), self.dlogsigma))
        self.sigma_index = int(len(self.allowed_sigma)/2)
        self.sigma = self.allowed_sigma[self.sigma_index]


    def load_data(self, file, verbose=False):
        """Load in the experimental restraints from a known
        file format."""

        df = pd.read_pickle(file)
        return df


    def add_restraint(self, restraint):
        """Add a new restraint data container (e.g. NMR_Chemicalshift()) to the list.

        :param list restraint:
        """

        self.restraints.append(restraint)

    def compute_sse(self, debug=False):
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        # Does the restraint child class contain any gamma information?
        if hasattr(self, 'allowed_gamma'):
            self.sse = np.array([0.0 for gamma in self.allowed_gamma])
            for g in range(len(self.allowed_gamma)):
                sse = 0.0
                N = 0.0
                for i in range(self.n):
                    gamma = self.allowed_gamma[g]
                    if self.use_log_normal_noe:
                        err = np.log(self.restraints[i].model/(gamma*self.restraints[i].exp))
                    else:
                        err = gamma*self.restraints[i].exp - self.restraints[i].model
                    sse += (self.restraints[i].weight * err**2.0)
                    N += self.restraints[i].weight
                self.sse[g] = sse
                self.Ndof = N

            if debug:
                for i in range(self.n):
                    print('---->', i, '%d'%self.restraints[i].i, end=' ')
                    print('      exp', self.restraints[i].exp, 'model', self.restraints[i].model)

        elif hasattr(self,'allowed_beta_c'):

            self.sse = np.zeros(  (len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                                                len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)))
            self.Ndof = 0.
            for i in range(self.n):
                err = self.restraints[i].model - self.restraints[i].exp
                self.sse += (self.restraints[i].weight * err**2.0)
                self.Ndof += self.restraints[i].weight

        else:
            sse = 0.0
            N = 0.0
            for i in range(self.n):
                if debug:
                    print('---->', i, '%d'%self.restraints[i].i, end=' ')
                    print('      exp', self.restraints[i].exp, 'model', self.restraints[i].model)

                err = self.restraints[i].model - self.restraints[i].exp
                sse += (self.restraints[i].weight*err**2.0)
                N += self.restraints[i].weight
            self.sse = sse
            self.Ndof = N
            if debug:
                print('self.sse', self.sse)


    def compute_neglog_exp_ref(self):
        """Uses the stored beta information (calculated across all structures)
        to calculate -log P_ref(observable[j]) for each observable j."""

        self.neglog_exp_ref = np.zeros(self.n)
        self.sum_neglog_exp_ref = 0.0
        for j in range(self.n):
            self.neglog_exp_ref[j] = np.log(self.betas[j])\
                    + self.restraints[j].model/self.betas[j]
            self.sum_neglog_exp_ref  += self.restraints[j].weight * self.neglog_exp_ref[j]

    def compute_neglog_gaussian_ref(self):
        """An alternative option for reference potential based on
        Gaussian distribution. (Ignoring constant terms) """

        self.neglog_gaussian_ref = np.zeros(self.n)
        self.sum_neglog_gaussian_ref = 0.0
        for j in range(self.n):
            self.neglog_gaussian_ref[j] = np.log(np.sqrt(2.0*np.pi))\
                    + np.log(self.ref_sigma[j]) + (self.restraints[j].model \
                    - self.ref_mean[j])**2.0/(2.0*self.ref_sigma[j]**2.0)
            self.sum_neglog_gaussian_ref += self.restraints[j].weight * self.neglog_gaussian_ref[j]



class Restraint_cs(Restraint):
    """A derived class of RestraintClass() for N chemical shift restraints."""

    _ext = ["H", "Ca", "N"]

    def __repr__(self):
        if self.extension is not None:
            return "<%s.Restraint_cs_%s>"%(str(__name__),str(self.extension))
        else:
            pass

    def init_restraint(self, data, energy, lam, extension, verbose=False):
        """Observable is prepped by loading in cs_N restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation"""

        self.extension = extension

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_cs_%s'%self.extension]

        # Reading the data from loading in filenames
        data = self.load_data(data)
        self.n = len(data.values)

        # Group by keys
        keys = ['atom_index1', 'exp', 'model']
        grouped_data = data[keys]
        for row in range(grouped_data.shape[0]):
            self.add_restraint(grouped_data.iloc[row])
            # N equivalent chemical shift should only get 1/N f the weight when
            #... computing chi^2 (not likely in this case but just in case we need it in the future)
            self.restraints[-1].weight = 1.0  #1.0/3.0 used in JCTC 2020 paper  # default is N=1
        self.compute_sse(debug=False)

    def compute_neglogP(self, index, parameters, parameter_indices, ln2pi):

        result = 0
        para,indices = parameters, parameter_indices
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(para[index][0])
        result += self.sse / (2.0*float(para[index][0])**2.0)
        result += (self.Ndof)/2.0*ln2pi  # for normalization
        if self.ref == "exp":
            result -= self.sum_neglog_exp_ref
        if self.ref == "gaussian":
            result -= self.sum_neglog_gaussian_ref
        return result


class Restraint_J(Restraint):
    """A derived class of RestraintClass() for J coupling constant."""

    _ext = ['J']

    def init_restraint(self, data, energy, lam, verbose=False):
        """Observable is prepped by loading in J coupling restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation"""

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_J']

        # Reading the data from loading in filenames
        data = self.load_data(data)
        self.n = len(data.values)

        # Group by keys
        keys = ['atom_index1', 'atom_index2', 'atom_index3', 'atom_index4',
                'exp', 'model', 'restraint_index']
        grouped_data = data[keys]
        for row in range(grouped_data.shape[0]):
            self.add_restraint(grouped_data.iloc[row])

        self.equivalency_groups = {}
        # Compile equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.restraints)):
            d = self.restraints[i]
            if d.restraint_index != None:
                if d.restraint_index not in self.equivalency_groups:
                    self.equivalency_groups[d.restraint_index] = []
                self.equivalency_groups[d.restraint_index].append(i)
        if verbose:
            print(f'grouped_data = {grouped_data}')
            print(f'self.restraints[0] = {self.restraints[0]}')
            print(f'self.equivalency_groups = {self.equivalency_groups}')
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.compute_sse(debug=False)


    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i].weight = 1.0/n

    def compute_neglogP(self, index, parameters, parameter_indices, ln2pi):

        result = 0
        para,indices = parameters, parameter_indices
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(para[index][0])
        result += self.sse / (2.0*float(para[index][0])**2.0)
        result += (self.Ndof)/2.0*ln2pi  # for normalization
        if self.ref == "exp":
            result -= self.sum_neglog_exp_ref
        if self.ref == "gaussian":
            result -= self.sum_neglog_gaussian_ref
        return result



class Restraint_noe(Restraint):
    """A derived class of Restraint() for noe distance restraints."""

    _ext = ['noe']

    def init_restraint(self, data, energy, lam, verbose=False,
            use_log_normal_noe=False, dloggamma=np.log(1.01),
            gamma_min=0.2, gamma_max=10.0):
        """Observable is prepped by loading in noe distance restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy f = beta*F of this conformation
        :param float dloggamma: Gamma is in log space
        :param float gamma_min: Minimum value of gamma
        :param float gamma_max: Maximum value of gamma"""

        # Store info about gamma^(-1/6) scaling parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
        self.gamma_index = int(len(self.allowed_gamma)/2)
        self.gamma = self.allowed_gamma[self.gamma_index]

        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_noe = use_log_normal_noe

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        #self.energy = energy
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma','allowed_gamma']
        self._parameters = ['sigma','gamma']
        self._parameter_indices = ['sigma_index','gamma_index']
        self._rest_type = ['sigma_noe','gamma']

        # Reading the data from loading in filenames
        data = self.load_data(data)
        self.n = len(data.values)

        # Group by keys
        keys = ['atom_index1', 'atom_index2', 'exp', 'model', 'restraint_index']
        grouped_data = data[keys]
        for row in range(grouped_data.shape[0]):
            self.add_restraint(grouped_data.iloc[row])

        self.equivalency_groups = {}
        for i in range(len(self.restraints)):
            d = self.restraints[i]
            if d.restraint_index != None:
                if d.restraint_index not in self.equivalency_groups:
                    self.equivalency_groups[d.restraint_index] = []
                self.equivalency_groups[d.restraint_index].append(i)
        if verbose:
            print(f'grouped_data = {grouped_data}')
            #print(f'self.restraints[0] = {self.restraints[0]}')
            print(f'self.equivalency_groups = {self.equivalency_groups}')
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.compute_sse(debug=False)

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                self.restraints[i].weight = 1.0/n

    def compute_neglogP(self, index, parameters, parameter_indices, ln2pi):

        result = 0
        para,indices = parameters, parameter_indices
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(para[index][0])
        result += self.sse[int(indices[index][1])] / (2.0*para[index][0]**2.0)
        result += (self.Ndof)/2.0*ln2pi  # for normalization
        if self.ref == "exp":
            result -= self.sum_neglog_exp_ref
        if self.ref == "gaussian":
            result -= self.sum_neglog_gaussian_ref
        return result




class Restraint_pf(Restraint):
    """A derived class of Restraint() for protection factor restraints."""

    _ext = ['pf']

    def init_restraint(self, lam, energy, data, precomputed=False,
            Ncs=None, Nhs=None,verbose=False, beta_c_min=0.05,beta_c_max=0.25,
            dbeta_c=0.01,beta_h_min=0.0,beta_h_max=5.2,dbeta_h=0.2,
            beta_0_min=-10.0,beta_0_max=0.0,dbeta_0=0.2,xcs_min=5.0,xcs_max=8.5,
            dxcs=0.5,xhs_min=2.0,xhs_max=2.7,dxhs=0.1,bs_min=15.0,bs_max=16.0,dbs=1.0):
        """Observable is prepped by loading in protection factor restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy f = beta*F of this conformation
        """

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        #self.energy = energy
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        if precomputed:
            self._nuisance_parameters = ['allowed_sigma']
            self._parameters = ['sigma']
            self._parameter_indices = ['sigma_index']
            self._rest_type = ['sigma_PF']
        else:
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
            #print "self.allowed_beta_c", len(self.allowed_beta_c), "self.allowed_beta_h", len(self.allowed_beta_h), "self.allowed_beta_0", len(self.allowed_beta_0), "self.allowed_xcs", len(self.allowed_xcs), "self.allowed_xhs", len(self.allowed_xhs), "self.allowed_bs", len(self.allowed_bs)
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

            self._nuisance_parameters = ['allowed_sigma','allowed_beta_c', 'allowed_beta_h',
                    'allowed_beta_0', 'allowed_xcs', 'allowed_xhs','allowed_bs']
            self._parameters = ['sigma','beta_c','beta_h','beta_0','xcs','xhs','bs']
            self._parameter_indices = ['sigma_index','beta_c_index','beta_h_index','beta_0_index','xcs_index','xhs_index','bs_index']
            self._rest_type = ['sigma_PF','beta_c','beta_h','beta_0','xcs','xhs','bs']


        # Reading the data from loading in filenames
        data = self.load_data(data)
        self.n = len(data.values)

        # Group by keys
        if precomputed:
            keys = ['atom_index1', 'exp', 'model']
        else:
            keys = ['atom_index1', 'exp']

        grouped_data = data[keys]
        for row in range(grouped_data.shape[0]):
            #TODO: test to make sure that this conditional works:
            if not precomputed:
                model = self.compute_PF_multi(self.Ncs[:,:,row], self.Nhs[:,:,row], debug=False)
                grouped_data.iloc[row]['model'] = model
            self.add_restraint(grouped_data.iloc[row])
        self.compute_sse(debug=False)


    def compute_PF(self, beta_c, beta_h, beta_0, Nc, Nh):
        """Calculate predicted (ln PF)

        :param np.array beta_c,beta_h,Nc,Nh: shape(nres, 2)
          array with columns <N_c> and <N_h> for each residue
        :return:   array of
          >>> <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
        """
        return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 '''


    def compute_PF_multi(self, Ncs_i, Nhs_i, debug=False):
        """Calculate predicted (ln PF)

        .. tip:: A near future application...

        :param np.array Ncs_i,Nhs_i:
          array with columns <N_c> and <N_h> for each residue
        :return:   array of
          >>> <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
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
        """Returns a multi-dimensional array of shape (tuple), with the 1D vector p along the specified axis,
           and tiled in all other dimensions.


        .. tip:: A near future application...
        :param np.array p: a 1D array to tile
        :param tuple shape: a tuple describing the shape of the returned array
        :var axis: the specified axis for p .  NOTE: len(p) must be equal to shape[axis]
        """

#        print 'shape', shape , 'axis', axis, 'p.shape', p.shape
        assert shape[axis] == len(p), "len(p) must be equal to shape[axis]!"

        otherdims = [shape[i] for i in range(len(shape)) if i!=axis]
        result = np.tile(p, tuple(otherdims+[1]))
#        print 'result.shape', result.shape
        last_axis = len(result.shape)-1
#        print 'last_axis, axis', last_axis, axis
        result2 = np.rollaxis(result, last_axis, axis)
#        print 'result2.shape', result2.shape
        return result2

    def tile_2D_multiaxis(self, q, shape, axes=None):
        """Returns a multi-dimensional array of shape (tuple), with the 2D vector p along the specified axis
           and tiled in all other dimensions.

        .. tip:: A near future application...
        :param np.array p: a 1D array to tile
        :param tuple shape: a tuple describing the shape of the returned array
        :var axis: the specified axis for p .  NOTE: len(p) must be equal to shape[axis]
        """

#        print 'shape', shape , 'axes', axes, 'q.shape', q.shape
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
#       print "self.nprotectionfactor", self.nprotectionfactor
        for j in range(self.n): # number of residues
#           print "self.protectionfactor_restraints[j]", self.protectionfactor_restraints[j]
            self.neglog_exp_ref[j] = np.maximum(-1.0*self.restraints[j].model, 0.0)
#           print "sum", sum(self.neglog_reference_priors_PF[j])
            self.sum_neglog_exp_ref  += self.restraints[j].weight * self.neglog_exp_ref[j]


    def compute_neglog_gaussian_ref_pf(self):
        self.neglog_gaussian_ref = np.zeros((self.n, len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0),
                                       len(self.allowed_xcs),    len(self.allowed_xhs),    len(self.allowed_bs)))
        self.sum_neglog_gaussian_ref = 0.
#       print "self.nprotectionfactor", self.nprotectionfactor
        for j in range(self.n): # number of residues
            self.neglog_gaussian_ref[j] = 0.5 * np.log(2.0*np.pi) + np.log(self.ref_sigma[j]) \
                      + (self.restraints[j].model - self.ref_mean[j])**2.0/(2.0*self.ref_sigma[j]**2.0)
            self.sum_neglog_gaussian_ref += self.restraints[j].weight * self.neglog_gaussian_ref[j]


    def compute_neglogP(self, index, parameters, parameter_indices, ln2pi):

        result = 0
        para,indices = parameters, parameter_indices
        # Use with log-spaced sigma values
        result += (self.Ndof)*np.log(para[index][0])
        result += self.sse[int(indices[index][1])][int(indices[index][2])][int(indices[index][3])][int(indices[index][4])][int(indices[index][5])][int(indices[index][6])] / (2.0*para[index][0]**2.0)
        if self.pf_prior is not None:
            result += self.pf_prior[int(indices[index][1])][int(indices[index][2])][int(indices[index][3])][int(indices[index][4])][int(indices[index][5])][int(indices[index][6])]
        else:
            result += self.sse / (2.0*float(para[index][0])**2.0)
        result += (self.Ndof)/2.0*ln2pi  # for normalization
        # Which reference potential was used for each restraint?
        if self.allowed_beta_c:
            if hasattr(s[index], 'sum_neglog_exp_ref'):
                if isinstance(self.sum_neglog_exp_ref, float):   # check if it is 0.0
                    result -= self.sum_neglog_exp_ref
                else:
                    result -= self.sum_neglog_exp_ref[int(indices[index][1])][int(indices[index][2])][int(indices[index][3])][int(indices[index][4])][int(indices[index][5])][int(indices[index][6])]
            if hasattr(s[index], 'sum_neglog_gaussian_ref'):
                if isinstance(self.sum_neglog_gaussian_ref, float):
                    result -= self.sum_neglog_gaussian_ref
                else:
                    result -= self.sum_neglog_gaussian_ref[int(indices[index][1])][int(indices[index][2])][int(indices[index][3])][int(indices[index][4])][int(indices[index][5])][int(indices[index][6])]
        else:
            if hasattr(s[index], 'sum_neglog_exp_ref'):
                result -= self.sum_neglog_exp_ref
            if hasattr(s[index], 'sum_neglog_gaussian_ref'):
                result -= self.sum_neglog_gaussian_ref
        return result





class Preparation(object):

    def __init__(self, nstates=0,  top=None, outdir="./"):
        """A parent class to prepare input files for BICePs calculation.

        :param str obs: type of experimental observables {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}
        :param int default=0 nstates: number of states
        :param str default=None indices: experimental observable index (*.txt file)
        :param str default=None exp_data: experimental measuremnets (*.txt file)
        :param str default=None top: topology file (*.pdb)
        """

        self.nstates = nstates
        self.topology = md.load(top).topology
        self.data = list()

    def write_DataFrame(self, filename, to="pickle", verbose=True):
        """Write Pandas DataFrame to user specified filetype."""

        #biceps.toolbox.mkdir(self.outdir)
        #columns = { self.keys[i] : self.header[i] for i in range(len(self.keys)) }
        #print(columns)
        if verbose:
            print('Writing %s as %s...'%(filename,to))
        df = pd.DataFrame(self.biceps_df)
        #dfOut = getattr(self.df.rename(columns=columns), "to_%s"%to)
        dfOut = getattr(df, "to_%s"%to)
        dfOut(filename)

    #TODO: needs to be checked
    def prep_cs(self, exp_data, model_data, indices, extension, outdir=None, verbose=False):
        """A method containing input/output methods for writing chemicalshift
        Restaint Files.

        exp (ppm)
        model (ppm)
        """

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'exp', 'model', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        self.ind = np.loadtxt(indices)
        self.ind = np.array(self.ind).astype(int)

        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i])
                dd['atom_index1'].append(a1)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.cs_%s"%(j, extension)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)



    def prep_noe(self, exp_data, model_data, indices, extension=None, outdir=None, verbose=False):
        """A method containing input/output methods for writing NOE
        Restaint Files.

        'exp' (A)
        'model' (A)
        """

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'atom_index2', 'res2', 'atom_name2', 'exp', 'model', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1, a2 = int(self.ind[i,0]), int(self.ind[i,1])
                dd['atom_index1'].append(a1)
                dd['atom_index2'].append(a2)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.noe"%(j)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)


    def prep_J(self, exp_data, model_data, indices, extension=None, outdir=None, verbose=False):
        """A method containing input/output methods for writing scalar coupling
        Restaint Files.

        'exp_J (Hz)
        'model_J (Hz)'
        """

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'atom_index2', 'res2', 'atom_name2', 'atom_index3', 'res3', 'atom_name3',
                'atom_index4', 'res4', 'atom_name4', 'exp',
                'model', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        if type(indices) is not str:
            self.ind = indices
        else:
            self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1, a2, a3, a4   = int(self.ind[i,0]), int(self.ind[i,1]), int(self.ind[i,2]), int(self.ind[i,3])
                dd['atom_index1'].append(a1);dd['atom_index2'].append(a2)
                dd['atom_index3'].append(a3);dd['atom_index4'].append(a4)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                dd['res3'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a3][0]))
                dd['atom_name3'].append(str([atom.name for atom in self.topology.atoms if atom.index == a3][0]))
                dd['res4'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a4][0]))
                dd['atom_name4'].append(str([atom.name for atom in self.topology.atoms if atom.index == a4][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.J"%(j)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)


    # TODO: Needs to be checked
    def prep_pf(self, exp_data, model_data=None, indices=None, extension=None, outdir=None, verbose=False):
        """A method containing input/output methods for writing protection factor
        Restaint Files."""

        if model_data:
            self.header = ('restraint_index', 'atom_index1', 'res1', 'exp','model', 'comments')
        else:
            self.header = ('restraint_index', 'atom_index1', 'res1','exp', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray or None:
            self.model_data = biceps.toolbox.get_files(model_data)
            if int(len(self.model_data)) != int(self.nstates):
                raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the\
                    number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i,0])
                dd['atom_index1'].append(a1)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                if model_data:
                    dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.pf"%(j)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)




if __name__ == "__main__":

    import doctest
    doctest.testmod()





