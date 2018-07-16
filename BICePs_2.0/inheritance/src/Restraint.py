##############################################################################
# Authors: Vincent Voelz, Yunhui Ge, Rob Raddi
# This file is used to initialize variables fir BICePs calculations.
# It is a parent class of each child class for different experimental observables.
##############################################################################

##############################################################################
# Imports
##############################################################################
import os, sys, glob
import numpy as np
import mdtraj
from KarplusRelation import * # Returns J-coupling values from dihedral angles
from toolbox import *

##############################################################################
# Code
##############################################################################

class Restraint(object):
    """The parent class of all Restraint() objects."""

    def __init__(self, PDB_filename, lam, free_energy, data=None,
            use_log_normal_noe=False, dloggamma=np.log(1.01), gamma_min=0.2,
            gamma_max=10.0):
        """Initialize the Restraint class.
        INPUTS
        PDB_filename        A topology file (*.pdb)
        lam        lambda value (between 0 and 1)
        free_energy     The (reduced) free energy f = beta*F of this conformation
        data            input data for BICePs (both model and exp)
        use_log_normal_distances    Not sure what's this...
        dloggamma    gamma is in log space
        gamma_min    min value of gamma
        gamma_max    max value of gamma
        """

        # Store restraint info
        self.restraints = []   # a list of data container objects for each restraint (e.g. NMR_Chemicalshift_Ca())
        self.n = 0
        self.sse = 0
        self.Ndof = None

        # used for exponential reference potential
        self.betas = None
        self.neglog_exp_ref = None
        self.sum_neglog_exp_ref = 0.0

        # used for Gaussian reference potential
        self.ref_sigma = None
        self.ref_mean = None
        self.neglog_gau_ref = None
        self.sum_neglog_gau_ref = 0.0

        #NOTE: This is where we appended the code from previous
        self.PDB_filename = PDB_filename
        self.data = data
        self.conf = mdtraj.load_pdb(PDB_filename)
        # Convert the coordinates from nm to Angstrom units
        self.conf.xyz = self.conf.xyz*10.0

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.free_energy = lam*free_energy

        # Store info about gamma^(-1/6) scaling  parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))

        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_noe = use_log_normal_noe

        # Create a KarplusRelation object
        self.karplus = KarplusRelation()


    def load_data(self, filename, verbose=False):
        """Load in the experimental data."""
        pass

    #def add_restraint(self, i, exp, model=None):
    def add_restraint(self, restraint):
        """Add a new restraint data container (e.g. NMR_Chemicalshift()) to the list."""

        self.restraints.append(restraint)
                #NMR_Chemicalshift(i, model, exp))
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



class NMR_Chemicalshift(object):
    """A data containter class to store a datum for NMR chemical shift information."""

    def __init__(self, i, model, exp):
        """Initialize the derived NMR_Chemicalshift class."""

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i
        # the model chemical shift in this structure (in ppm)
        self.model = model
        # the experimental chemical shift
        self.exp = exp
        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1



class NMR_Dihedral(object):
    """A data containter class to store a datum for NMR dihedral information."""

    def __init__(self, i, j, k, l, model, exp, model_angle,
            equivalency_index=None, ambiguity_index=None):

        # Atom indices from the Conformation() defining this dihedral
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        # the model distance in this structure (in Angstroms)
        self.model = model
        # the experimental J-coupling constant
        self.exp = exp
        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index
        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1
        # the index of the ambiguity group (i.e. some groups distances have
        # distant values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index


class NMR_Distance(object):
    """A class to store NMR noe information."""

    def __init__(self, i, j, model, exp, equivalency_index=None):

        # Atom indices from the Conformation() defining this noe
        self.i = i
        self.j = j

        # the model noe in this structure (in Angstroms)
        self.model = model

        # the experimental NOE noe (in Angstroms)
        self.exp = exp

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent noe should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1


class NMR_Protectionfactor(object):
    """A class to store NMR protection factor information."""

    def __init__(self, i, model, exp):
        # Atom indices from the Conformation() defining this protection factor
        self.i = i
        # the model protection factor in this structure (in ???)
        self.model = model
        # the experimental protection factor
        self.exp = exp
        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1











