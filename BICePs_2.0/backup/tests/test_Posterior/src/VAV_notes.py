##############################################################################
# Authors: Yunhui Ge
# Contributors: Vincent Voelz, Rob Raddi
# This file is used to initialize variables for chemical shifts (CA) in BICePs and prepare fuctions for compute necessary quantities for posterior sampling.
##############################################################################


##############################################################################
# Imports
##############################################################################


import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_cs import *    # Class - creates Chemical shift restraint file

##############################################################################
# Code
##############################################################################

class RestraintClass(object):
    """The parent class of all RestraintClass() objects."""

   def __init__(self):
       """Initialize the Restraint Class"

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

    def load_data(self, filename, verbose=False):
        """Load in the experimental data."""
        pass
        
    def add_restraint(self, i, exp, model=None):
        """Add a new restraint data container (e.g. NMR_Chemicalshift()) to the list."""
        pass
    
    def compute_sse(self, debug=False):    #GYCa
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

    def compute_neglog_exp_ref(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(observable[j]) for each observable j."""

        # print 'self.betas', self.betas

        self.neglog_exp_ref = np.zeros(self.n)
        self.sum_neglog_exp_ref = 0.
        for j in range(self.n):
            self.neglog_exp_ref[j] = np.log(self.betas[j]) + self.restraints[j].model/self.betas[j]
            self.sum_neglog_exp_ref  += self.restraints[j].weight * self.neglog_exp_ref[j]

    def compute_neglog_gau_ref(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.neglog_gau_ref = np.zeros(self.n) 
        self.sum_neglog_gau_ref_ = 0.
        for j in range(self.n):
            self.neglog_gau_ref[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma[j]) + (self.restraints[j].model - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
            self.sum_neglog_gau_ref += self.restraints[j].weight * self.neglog_gau_ref[j]
        
class CS_CAlpha_RestraintClass(RestraintClass):
    """A derived class of RestraintClass() for C_alpha chemical shift restraints."""
    

    def load_data(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .cs file format.
        """

        # Read in the lines of the cs data file
        b = prep_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line(line) )  # [restraint_index, atom_index1, res1, atom_name1, cs]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_cs_Ca, model_cs_Ca  = entry[0], entry[1], entry[4], entry[5]
            self.add_restraint(i, exp_cs_Ca, model_cs_Ca)


        self.compute_sse()



    def add_cs_Ca_restraint(self, i, exp_cs_Ca, model_cs_Ca=None):
        """Add a cs NMR_Chemicalshift() object to the list."""

        self.cs_Ca_restraints.append( NMR_Chemicalshift_Ca(i, model_cs_Ca, exp_cs_Ca))

        self.ncs_Ca += 1

    def compute_sse_cs_Ca(self, debug=False):    #GYCa
        """Returns the (weighted) sum of squared errors for chemical shift values"""

        sse_Ca = 0.0
        N_Ca = 0.0
        for i in range(self.ncs_Ca):
		if debug:
               		print '---->', i, '%d'%self.cs_Ca_restraints[i].i,
               		print '      exp', self.cs_Ca_restraints[i].exp_cs_Ca, 'model', self.cs_Ca_restraints[i].model_cs_Ca

                err_Ca=self.cs_Ca_restraints[i].model_cs_Ca - self.cs_Ca_restraints[i].exp_cs_Ca
                sse_Ca += (self.cs_Ca_restraints[i].weight*err_Ca**2.0)
                N_Ca += self.cs_Ca_restraints[i].weight
        self.sse_cs_Ca = sse_Ca
        self.Ndof_cs_Ca = N_Ca
        if debug:
            print 'self.sse_cs_Ca', self.sse_cs_Ca

    def compute_neglog_exp_ref_Ca(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_exp_ref_Ca = np.zeros(self.ncs_Ca)
        self.sum_neglog_exp_ref_Ca = 0.
        for j in range(self.ncs_Ca):
            self.neglog_exp_ref_Ca[j] = np.log(self.betas_Ca[j]) + self.cs_Ca_restraints[j].model_cs_Ca/self.betas_Ca[j]
            self.sum_neglog_exp_ref_Ca  += self.cs_Ca_restraints[j].weight * self.neglog_exp_ref_Ca[j]

    def compute_neglog_gau_ref_Ca(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.neglog_gau_ref_Ca = np.zeros(self.ncs_Ca)
        self.sum_neglog_gau_ref_Ca = 0.
        for j in range(self.ncs_Ca):
            self.neglog_gau_ref_Ca[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ca[j]) + (self.cs_Ca_restraints[j].model_cs_Ca - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
            self.sum_neglog_gau_ref_Ca += self.cs_Ca_restraints[j].weight * self.neglog_gau_ref_Ca[j]



class NMR_Chemicalshift_Ca(object):        #GYH
    """A data containter class to store a datum for NMR chemical shift information."""

    # __init__:{{{
    def __init__(self, i, model_cs_Ca, exp_cs_Ca):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_cs_Ca = model_cs_Ca

        # the experimental chemical shift
        self.exp_cs_Ca = exp_cs_Ca

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1






######


my_rest = []
my_rest.append( CS_CAlpha_RestraintClass() )
my_rest.append( CS_N_RestraintClass() )

sum_sse = 0.0
for i in range(len(my_rest)):
  sum_sse += my_rest[i].sse




