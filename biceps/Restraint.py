# -*- coding: utf-8 -*-
import os, sys, glob
import numpy as np
import mdtraj
from biceps.KarplusRelation import * # Returns J-coupling values from dihedral angles
from biceps.toolbox import *
from biceps.prep_cs import *    # Creates Chemical shift restraint file
from biceps.prep_J import *     # Creates J-coupling const. restraint file
from biceps.prep_noe import *   # Creates NOE (Nuclear Overhauser effect) restraint file
from biceps.prep_pf import *    # Prepare functions for protection factors restraint file
import biceps.Observable as Observable

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
        self.conf = mdtraj.load_pdb(PDB_filename)

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


    def load_data(self, prep, verbose=False):
        """Load in the experimental chemical shift restraints from a known
        file format.

        :param file prep: prep the input data
        """

        # Read in the lines of the cs data file
        read = prep
        if verbose:
            print(read.lines)
        data = []
        for line in read.lines:
            data.append( read.parse_line(line) )
        self.data = data


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


class Restraint_cs_Ca(Restraint):
    """A derived class of RestraintClass() for C_alpha chemical shift restraints."""

    _ext = 'cs_Ca'

    def prep_observable(self,data,energy,lam,verbose=False):
        """Observable is prepped by loading in C_alpha restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation

        >>> f = beta*F
        """

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        ##self.energy = energy
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_cs_Ca']

        # Reading the data from loading in filenames
        read = prep_cs(filename=data)
        self.load_data(read)

        # Add the chemical shift restraints
        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print('Loaded from', data, ':')
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs =  Observable.NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print(entry)
            self.n += 1
        self.compute_sse(debug=False)


class Restraint_cs_H(Restraint):
    """A derived class of RestraintClass() for H chemical shift restraints."""

    _ext = 'cs_H'

    def prep_observable(self,data,energy,lam,verbose=False):
        """Observable is prepped by loading in cs_H restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation
        """

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        #self.energy = energy
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_cs_H']

        # Reading the data from loading in filenames
        read = prep_cs(filename=data)
        self.load_data(read)
        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print('Loaded from', data, ':')
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = Observable.NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print(entry)
            self.n += 1
        self.compute_sse(debug=False)


class Restraint_cs_Ha(Restraint):
    """A derived class of RestraintClass() for Ha chemical shift restraints."""

    _ext = 'cs_Ha'

    def prep_observable(self,data,energy,lam,verbose=False):
        """Observable is prepped by loading in cs_Ha restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation
        """

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        #self.energy = energy
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        self._nuisance_parameters = ['allowed_sigma']
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_cs_Ha']

        # Reading the data from loading in filenames
        read = prep_cs(filename=data)
        self.load_data(read)
        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print('Loaded from', data, ':')
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = Observable.NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print(entry)
            self.n += 1
        self.compute_sse(debug=False)


class Restraint_cs_N(Restraint):
    """A derived class of RestraintClass() for N chemical shift restraints."""

    _ext = 'cs_N'

    def prep_observable(self,data,energy,lam,verbose=False):
        """Observable is prepped by loading in cs_N restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation"""

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        #self.energy = energy
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_cs_N']

        # Reading the data from loading in filenames
        read = prep_cs(filename=data)
        self.load_data(read)
        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print('Loaded from', data, ':')
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = Observable.NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print(entry)
            self.n += 1
        self.compute_sse(debug=False)


class Restraint_J(Restraint):
    """A derived class of RestraintClass() for J coupling constant."""

    _ext = 'J'

    def prep_observable(self,data,energy,lam,verbose=False):
        """Observable is prepped by loading in J coupling restraints.

        :param str data: Experimental data file
        :param float lam: Lambda value (between 0 and 1)
        :param float energy: The (reduced) free energy of this conformation"""

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        #self.energy = energy
        #print(type(energy))
        self.energy = np.float128(lam*energy)
        self.Ndof = None
        # Private variables to store specific restraint attributes in a list
        self._nuisance_parameters = ['allowed_sigma']
        self._parameters = ['sigma']
        self._parameter_indices = ['sigma_index']
        self._rest_type = ['sigma_J']

        # Reading the data from loading in filenames
        read = prep_J(filename=data)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, j, k, l, exp, model  = entry[0], entry[1],\
                    entry[4], entry[7], entry[10], entry[13], entry[14]

            Obs = Observable.NMR_Dihedral(i,j,k,l,exp,model,
                    equivalency_index=restraint_index)
            self.add_restraint(Obs)
            if verbose:
                print(entry)
            self.n += 1

        self.equivalency_groups = {}

        # Compile equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.restraints)):
            if 'NMR_Dihedral' in self.restraints[i].__str__():
                d = self.restraints[i]
                if d.equivalency_index != None:
                    if d.equivalency_index not in self.equivalency_groups:
                        self.equivalency_groups[d.equivalency_index] = []
                    self.equivalency_groups[d.equivalency_index].append(i)

        if verbose:
            print('self.equivalency_groups', self.equivalency_groups)
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.compute_sse(debug=False)


    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                if 'NMR_Dihedral' in self.restraints[i].__str__():
                    self.restraints[i].weight = 1.0/n



class Restraint_noe(Restraint):
    """A derived class of Restraint() for noe distance restraints."""

    _ext = 'noe'

    def prep_observable(self, data, energy, lam, verbose=False,
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
        read = prep_noe(filename=data)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, j, exp, model = entry[0], entry[1], entry[4], entry[7], entry[8]
           # ri = self.conf.xyz[0,i,:]
           # rj = self.conf.xyz[0,j,:]
           # dr = rj-ri
           # model = np.dot(dr,dr)**0.5
            Obs = Observable.NMR_Distance(i, j, exp, model, equivalency_index=restraint_index)
            self.add_restraint(Obs)
            if verbose:
                print(entry)
            self.n += 1

        self.equivalency_groups = {}

        #FIXME: equivalency groups & adjust weights is unfinished and edits need to be made

        # Compile equivalency_groups from the list of NMR_Distance() objects
        for i in range(len(self.restraints)):
            if 'NMR_Distance' in self.restraints[i].__str__():
                d = self.restraints[i]
                if d.equivalency_index != None:
                    if d.equivalency_index not in self.equivalency_groups:
                        self.equivalency_groups[d.equivalency_index] = []
                    self.equivalency_groups[d.equivalency_index].append(i)

        if verbose:
            print('self.equivalency_groups', self.equivalency_groups)

        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.compute_sse(debug=False)

    def adjust_weights(self):
        """Adjust the weights of distance restraints based on
        their equivalency group."""

        for group in list(self.equivalency_groups.values()):
            n = float(len(group))
            for i in group:
                if 'NMR_Distance' in self.restraints[i].__str__():
                    self.restraints[i].weight = 1.0/n



class Restraint_pf(Restraint):
    """A derived class of Restraint() for protection factor restraints."""

    _ext = 'pf'

    def prep_observable(self,lam, energy, data, precomputed_pf = False,
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
        if precomputed_pf:
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
            #sys.exit()
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
        read = prep_pf(filename=data)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print('Loaded from', data, ':')

        self.nObs = len(self.data)
        if precomputed_pf:
            for entry in self.data:
                restraint_index, i, exp, model  = entry[0], entry[0], entry[3], entry[4]
                Obs = Observable.NMR_Protectionfactor(i, exp, model)
                self.add_restraint(Obs)
                if verbose:
                    print(entry)
                self.n += 1
        else:
            for entry in self.data:
                restraint_index, i, exp  = entry[0], entry[0], entry[3]
                model = self.compute_PF_multi(self.Ncs[:,:,i], self.Nhs[:,:,i], debug=False)
                Obs = Observable.NMR_Protectionfactor(i, exp, model)
                self.add_restraint(Obs)
                if verbose:
                    print(entry)
                self.n += 1
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




