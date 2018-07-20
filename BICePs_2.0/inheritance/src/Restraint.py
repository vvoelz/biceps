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
from prep_cs import *    # Class - creates Chemical shift restraint file
from prep_J import *     # Creates J-coupling const. restraint file
from prep_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from prep_pf import *    # Class - prepare functions for protection factors restraint file
from Observable import * #

##############################################################################
# TODO:
##############################################################################
'''
 place all the correct initialization variables inside each of the children
Complete the remaining children - U stopped at cs_N


'''

##############################################################################
# Code
##############################################################################

class Restraint(object):
    """The parent class of all Restraint() objects."""

    def __init__(self, PDB_filename):
        """Initialize the Restraint class.
        INPUTS
        PDB_filename        A topology file (*.pdb)
        data            input data for BICePs (both model and exp)"""

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
        self.neglog_gau_ref = None
        self.sum_neglog_gau_ref = 0.0

        self.n = 0 # Initialize the overall restraint count


#    def load_data(self, filename, verbose=False):
#        """Load in the experimental data."""
#        pass
    def load_data(self, prep, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format."""

        # Read in the lines of the cs data file
        read = prep
        if verbose:
                print read.lines
        data = []
        for line in read.lines:
                data.append( read.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, cs]
        self.data = data






    def add_restraint(self, restraint):
        """Add a new restraint data container (e.g. NMR_Chemicalshift()) to the list."""

        self.restraints.append(restraint)
        self.n += 1

    #NOTE: try to merge "compute_sse" and "compute_sse_dihedrals"
    def compute_sse(self, debug=False):
        """Returns the (weighted) sum of squared errors for chemical shift values"""

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
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(observable[j]) for each observable j."""

        # print 'self.betas', self.betas
        self.neglog_exp_ref = np.zeros(self.n)
        self.sum_neglog_exp_ref = 0.
        for j in range(self.n):
            self.neglog_exp_ref[j] = np.log(self.betas[j]) + self.restraints[j].model/self.betas[j]
            self.sum_neglog_exp_ref  += self.restraints[j].weight * self.neglog_exp_ref[j]

    def compute_neglog_gau_ref(self):
        """An alternative option for reference potential based on Gaussian distribution"""
        self.neglog_gau_ref = np.zeros(self.n)
        self.sum_neglog_gau_ref_ = 0.
        for j in range(self.n):
            self.neglog_gau_ref[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma[j]) + (self.restraints[j].model - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
            self.sum_neglog_gau_ref += self.restraints[j].weight * self.neglog_gau_ref[j]


    def exp_uncertainty(self,dlogsigma=np.log(1.02),sigma_min=0.05,
            sigma_max=20.0):

        # Initialize values for sigma (std of experimental uncertainty)
        self.dlogsigma = dlogsigma  # stepsize in log(sigma) - i.e. grow/shrink multiplier
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.allowed_sigma = np.exp(np.arange(np.log(self.sigma_min), np.log(self.sigma_max), self.dlogsigma))
        self.sigma_index = len(self.allowed_sigma)/2
        self.sigma = self.allowed_sigma[self.sigma_index]





"""
        lam        lambda value (between 0 and 1)
        free_energy     The (reduced) free energy f = beta*F of this conformation
        use_log_normal_distances    Not sure what's this...
        dloggamma    gamma is in log space
        gamma_min    min value of gamma
        gamma_max    max value of gamma
"""


###########################################################################
# Children
###########################################################################

class Restraint_cs_Ca(Restraint):
    """A derived class of RestraintClass() for C_alpha chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):

        self.n = 0
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        # Add the chemical shift restraints
        if verbose:
            print 'Loaded from', filename, ':'
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            Obs = NMR_Chemicalshift(i, exp, model)
            self.add_restraint(Obs)
            if verbose:
                print entry
        self.compute_sse(debug=True)
        self.n += 1


class Restraint_cs_H(Restraint):
    """A derived class of RestraintClass() for H chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):

        self.n = 0
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        # add the chemical shift restraints
        if verbose:
            print 'Loaded from', filename, ':'
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Chemicalshift(i, exp, model))
            if verbose:
                print entry
        self.compute_sse(debug=True)
        self.n += 1



class Restraint_cs_Ha(Restraint):
    """A derived class of RestraintClass() for Ha chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):

        self.n = 0
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        # add the chemical shift restraints
        if verbose:
            print 'Loaded from', filename, ':'
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Chemicalshift(i, exp, model))
            if verbose:
                print entry
        self.compute_sse(debug=True)
        self.n += 1



class Restraint_cs_N(Restraint):
    """A derived class of RestraintClass() for N chemical shift restraints."""

    def prep_observable(self,filename,free_energy,lam,verbose=False):

        self.n = 0
        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.lam = lam
        self.free_energy = free_energy
        self.free_energy = lam*free_energy
        self.Ndof = None

        # Initialize the experimental uncertainties
        self.exp_uncertainty()

        # Reading the data from loading in filenames
        read = prep_cs(filename=filename)
        self.load_data(read)

        # add the chemical shift restraints
        if verbose:
            print 'Loaded from', filename, ':'
        for entry in self.data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Chemicalshift(i, exp, model))
            if verbose:
                print entry
        self.compute_sse(debug=True)
        self.n += 1





#NOTE: this derived class is not yet completed and is more difficult
class Restraint_J(Restraint):
    """A derived class of RestraintClass() for J coupling constant."""

    def load_data(self, filename, verbose=False):
        """Load in the experimental Jcoupling constant restraints from a .Jcoupling file format."""

        # Read in the lines of the biceps data file
        b = prep_J(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, atom_index3, res3, atom_name3, atom_index4, res4, atom_name4, J_coupling(Hz)]
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        ### Jcoupling ###
        # the equivalency indices for Jcoupling are in the first column of the *.Jcoupling file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'distance equivalency_indices', equivalency_indices
        # add the Jcoupling restraints
        for entry in data:
            restraint_index, i, j, k, l, exp, karplus  = entry[0], entry[1], entry[4], entry[7], entry[10], entry[13], entry[14]

            # if the modeled Jcoupling value is not specified, compute it from the
            # angle corresponding to the conformation, and the Karplus relation
            ri, rj, rk, rl = [self.conf.xyz[0,x,:] for x in [i, j, k, l]]
            model_angle = self.dihedral_angle(ri,rj,rk,rl)
            model = self.karplus.J(model_angle, "Karplus_HH")

        self.add_restraint(NMR_Dihedral(i,j,k,l,model,exp,model_angle,
            equivalency_index=equivalency_index))
        self.n += 1
        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on their equivalency group."""

        for group in self.dihedral_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.restraints[i].weight = 1.0/n








class Restraint_noe(Restraint):
    """A derived class of Restraint() for noe distance restraints."""

    def load_data_noe(self, filename, verbose=False):
        """Load in the experimental NOE noe restraints from a .noe file format."""



        use_log_normal_noe=False
        dloggamma=np.log(1.01)
        gamma_min=0.2
        gamma_max=10.0


        # Store info about gamma^(-1/6) scaling  parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
        self.gamma_index = len(self.allowed_gamma)/2
        self.gamma = self.allowed_gamma[self.gamma_index]


        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_noe = use_log_normal_noe




        # Read in the lines of the biceps data file
        b = prep_noe(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, noe]
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        # the equivalency indices for noe are in the first column of the *.biceps file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'noe equivalency_indices', equivalency_indices
        # add the noe restraints
        for entry in data:
            restraint_index, i, j, exp, model = entry[0], entry[1], entry[4], entry[7], entry[8]
            ri = self.conf.xyz[0,i,:]
            rj = self.conf.xyz[0,j,:]
            dr = rj-ri
            model = np.dot(dr,dr)**0.5
            self.add_restraint(NMR_Distance(i, j, model, exp,
                equivalency_index=equivalency_index))
        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()
        self.n += 1


class Restraint_pf(Restraint):
    """A derived class of Restraint() for protection factor restraints."""

    def load_data_pf(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format."""

        # Read in the lines of the protection factors data file
        b = prep_pf(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_pf(line) )
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        # add the protection factors restraints
        for entry in data:
            restraint_index, i, exp, model  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(NMR_Protectionfactor(i, model, exp))

        self.compute_sse()
        self.n += 1



class Restraint_pf_spec(Restraint):
    """A derived class of Restraint() for protection factor spec restraints."""

    def load_data_pf(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format."""

        # Read in the lines of the chemicalshift data file
        b = prep_pf(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry
        ### distances ###
        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]
        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp, model  = entry[0], entry[0], entry[3]
            model = self.compute_PF_multi(self.Ncs[:,:,i], self.Nhs[:,:,i], debug=True)
            self.add_restraint(NMR_Protectionfactor(i, model, exp))
        self.npf += 1
        self.compute_sse()


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
           and tiled in all other dimenions.

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
           and tiled in all other dimenions.

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



