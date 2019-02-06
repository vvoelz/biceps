##############################################################################
# Authors: Vincent Voelz, Yunhui Ge, Rob Raddi
# This file is used to initialize variables fir BICePs calculations.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
import mdtraj
from KarplusRelation import * # Returns J-coupling values from dihedral angles
from toolbox import *
from prep_cs import *    # Creates Chemical shift restraint file
from prep_J import *     # Creates J-coupling const. restraint file
from prep_noe import *   # Creates NOE (Nuclear Overhauser effect) restraint file
from prep_pf import *    # Prepare functions for protection factors restraint file
from Observable import * # Containers for experimental observables

##############################################################################
# Code
##############################################################################
class Restraint(object):
    """The parent class of all Restraint() objects."""

    def __init__(self, PDB_filename, ref, use_global_ref_sigma=True):
        """Initialize the Restraint class.

        INPUTS
        ------

        PDB_filename        A topology file (*.pdb)
        data            input data for BICePs (both model and exp)
        ref           Reference potential.
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
        self.see = None

        # Storing the reference potential
        self.ref = ref

    def load_data(self, prep, verbose=False):
        """Load in the experimental chemical shift restraints from a known
        file format."""

        # Read in the lines of the cs data file
        read = prep
        if verbose:
            print read.lines
        data = []
        for line in read.lines:
            data.append( read.parse_line(line) )
        self.data = data


    def add_restraint(self, restraint):
        """Add a new restraint data container (e.g. NMR_Chemicalshift()) to the list."""

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
                    print '---->', i, '%d'%self.restraints[i].i,
                    print '      exp', self.restraints[i].exp, 'model', self.restraints[i].model

                print 'self.sse', self.sse

        else:
            sse = 0.0
            N = 0.0
            for i in range(self.n):
                if debug:
                    print '---->', i, '%d'%self.restraints[i].i,
                    print '      exp', self.restraints[i].exp, 'model', self.restraints[i].model

                err = self.restraints[i].model - self.restraints[i].exp
                sse += (self.restraints[i].weight*err**2.0)
                N += self.restraints[i].weight
            self.sse = sse
            self.Ndof = N
            if debug:
                print 'self.sse', self.sse


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

    def exp_uncertainty(self, dlogsigma=np.log(1.02), sigma_min=0.05, sigma_max=20.0):
        """Initialize values for Std. deviation of experimental
        observables, sigma.

        Parameters
        ----------

        dlogsigma - step size in log(sigma) - i.e. grow/shrink multiplier
        sigma_min - minimum value of sigma
        sigma_max - maximum value of sigma """

        self.dlogsigma = dlogsigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.allowed_sigma = np.exp(np.arange(np.log(self.sigma_min),
            np.log(self.sigma_max), self.dlogsigma))
        self.sigma_index = len(self.allowed_sigma)/2
        self.sigma = self.allowed_sigma[self.sigma_index]


###########################################################################
# Children
###########################################################################
class Restraint_cs_Ca(Restraint):
    """A derived class of RestraintClass() for C_alpha chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in C_alpha restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""


        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        # Add the chemical shift restraints
        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print 'Loaded from', filename, ':'
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs =  NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1
        self.compute_sse(debug=True)


class Restraint_cs_H(Restraint):
    """A derived class of RestraintClass() for H chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in cs_H restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""


        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print 'Loaded from', filename, ':'
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1
        self.compute_sse(debug=True)


class Restraint_cs_Ha(Restraint):
    """A derived class of RestraintClass() for Ha chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in cs_Ha restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""


        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print 'Loaded from', filename, ':'
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1
        self.compute_sse(debug=True)

class Restraint_cs_N(Restraint):
    """A derived class of RestraintClass() for N chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in cs_N restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""


        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print 'Loaded from', filename, ':'
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1
        self.compute_sse(debug=True)



class Restraint_J(Restraint):
    """A derived class of RestraintClass() for J coupling constant."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in J coupling restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""


        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_J(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, j, k, l, exp, karplus  = entry[0], entry[1],\
                    entry[4], entry[7], entry[10], entry[13], entry[14]

            # if the modeled Jcoupling value is not specified, compute it from the
            # angle corresponding to the conformation, and the Karplus relation
            ri, rj, rk, rl = [self.conf.xyz[0,x,:] for x in [i, j, k, l]]
            model_angle = self.dihedral_angle(ri,rj,rk,rl)
            model = self.karplus.J(model_angle, "Karplus_HH")
            Obs = NMR_Dihedral(i,j,k,l,exp,model,model_angle,
                    equivalency_index=restraint_index)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1

        self.equivalency_groups = {}

        # Compile equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.restraints)):
            if 'NMR_Dihedral' in self.restraints[i].__str__():
                d = self.restraints[i]
                if d.equivalency_index != None:
                    if not self.equivalency_groups.has_key(d.equivalency_index):
                        self.equivalency_groups[d.equivalency_index] = []
                        self.equivalency_groups[d.equivalency_index].append(i)

        if verbose:
            print 'self.equivalency_groups', self.equivalency_groups
        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.compute_sse(debug=True)


    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on
        their equivalency group."""

        for group in self.equivalency_groups.values():
            n = float(len(group))
            for i in range(len(self.restraints)):
                if 'NMR_Dihedral' in self.restraints[i].__str__():
                    self.restraints[i].weight = 1.0/n



class Restraint_noe(Restraint):
    """A derived class of Restraint() for noe distance restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False,
            use_log_normal_noe=False,dloggamma=np.log(1.01),
            gamma_min=0.2,gamma_max=10.0):
        """Observable is prepped by loading in noe distance restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation
        dloggamma    - Gamma is in log space
        gamma_min    - Minimum value of gamma
        gamma_max    - Maximum value of gamma"""

        # Store info about gamma^(-1/6) scaling parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
        self.gamma_index = len(self.allowed_gamma)/2
        self.gamma = self.allowed_gamma[self.gamma_index]

        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_noe = use_log_normal_noe

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma','allowed_gamma']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_noe(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, j, exp, model = entry[0], entry[1], entry[4], entry[7], entry[8]
            ri = self.conf.xyz[0,i,:]
            rj = self.conf.xyz[0,j,:]
            dr = rj-ri
            model = np.dot(dr,dr)**0.5
            Obs = NMR_Distance(i, j, exp, model, equivalency_index=restraint_index)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1

        self.equivalency_groups = {}

        # Compile equivalency_groups from the list of NMR_Distance() objects
        for i in range(len(self.restraints)):
            if 'NMR_Distance' in self.restraints[i].__str__():
                d = self.restraints[i]
                if d.equivalency_index != None:
                    if not self.equivalency_groups.has_key(d.equivalency_index):
                        self.equivalency_groups[d.equivalency_index] = []
                        self.equivalency_groups[d.equivalency_index].append(i)

        if verbose:
            print 'self.equivalency_groups', self.equivalency_groups

        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()
        self.compute_sse(debug=True)

    def adjust_weights(self):
        """Adjust the weights of distance restraints based on
        their equivalency group."""

        for group in self.equivalency_groups.values():
            n = float(len(group))
            for i in range(len(self.restraints)):
                if 'NMR_Distance' in self.restraints[i].__str__():
                    self.restraints[i].weight = 1.0/n



class Restraint_pf(Restraint):
    """A derived class of Restraint() for protection factor restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in protection factor restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""


        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None
        self.nuisance_parameters = ['allowed_sigma','beta_c', 'beta_h',
                'beta_0', 'Nc', 'Nh']

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_pf(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print 'Loaded from', filename, ':'

        self.nObs = len(self.data)
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = NMR_Protectionfactor(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1

        self.compute_sse(debug=True)




class Restraint_pf_spec(Restraint):
    """A derived class of Restraint() for protection factor spec restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):
        """Observable is prepped by loading in protection factor spec. restraints.

        Parameters
        ----------

        filename     - Experimental data file
        lam          - Lambda value (between 0 and 1)
        free_energy  - The (reduced) free energy f = beta*F of this conformation"""



        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_pf(filename=filename)
        self.load_data(read)

        self.n = 0

        # Extract the data corresponding to an observable and add a the restraint
        if verbose:
            print 'Loaded from', filename, ':'

        self.nObs = len(self.data)
        for entry in self.data:
        ### distances ###
        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]
        # add the chemical shift restraints
            restraint_index, i, exp, model  = entry[0], entry[0], entry[3]
            model = self.compute_PF_multi(self.Ncs[:,:,i], self.Nhs[:,:,i], debug=True)
            Obs = NMR_Protectionfactor(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
            self.n += 1

        self.compute_sse(debug=True)



    def compute_PF(self, beta_c, beta_h, beta_0, Nc, Nh):
        """Calculate predicted (ln PF)
        INPUT    (nres, 2) array with columns <N_c> and <N_h> for each residue,
        OUTPUT   array of <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
        """
        return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 '''

    def compute_PF_multi(self, Ncs_i, Nhs_i, debug=False):
        """Calculate predicted (ln PF)
        INPUT    (nres, 2) array with columns <N_c> and <N_h> for each residue,
        OUTPUT   array of <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
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
            print 'nuisance_shape', nuisance_shape
            print 'beta_c.shape', beta_c.shape
            print 'beta_h.shape', beta_h.shape
            print 'beta_0.shape', beta_0.shape
            print 'Nc.shape', Nc.shape
            print 'Nh.shape', Nh.shape

        return beta_c * Nc + beta_h * Nh + beta_0

    def tile_multiaxis(self, p, shape, axis=None):
        """Returns a multi-dimensional array of shape (tuple), with the 1D vector p along the specified axis,
           and tiled in all other dimensions.

        INPUT
        p       a 1D array to tile
        shape   a tuple describing the shape of the returned array
        axis    the specified axis for p .  NOTE: len(p) must be equal to shape[axis]
        """


        print 'shape', shape , 'axis', axis, 'p.shape', p.shape
        assert shape[axis] == len(p), "len(p) must be equal to shape[axis]!"

        otherdims = [shape[i] for i in range(len(shape)) if i!=axis]
        result = np.tile(p, tuple(otherdims+[1]))
        print 'result.shape', result.shape
        last_axis = len(result.shape)-1
        print 'last_axis, axis', last_axis, axis
        result2 = np.rollaxis(result, last_axis, axis)
        print 'result2.shape', result2.shape
        return result2

    def tile_2D_multiaxis(self, q, shape, axes=None):
        """Returns a multi-dimensional array of shape (tuple), with the 2D vector p along the specified axis
           and tiled in all other dimensions.

        INPUT
        q       a 2D array to tile
        shape   a tuple describing the shape of the returned array
        axes    a list of two specified axes   NOTE: q.shape must be equal to (shape[axes[0]],shape[axes[1]])
        """

        print 'shape', shape , 'axes', axes, 'q.shape', q.shape
        assert (shape[axes[0]],shape[axes[1]]) == q.shape, "q.shape must be equal to (shape[axes[0]],shape[axes[1]])"

        otherdims = [shape[i] for i in range(len(shape)) if i not in axes]
        result = np.tile(q, tuple(otherdims+[1,1]))
        last_axis = len(result.shape)-1
        next_last_axis = len(result.shape)-2
        result2 = np.rollaxis(result, next_last_axis, axes[0])
        return np.rollaxis( result2, last_axis, axes[1])



