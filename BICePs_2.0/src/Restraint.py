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
#from msmbuilder import Conformation
import mdtraj
import yaml                       # Yet Another Multicolumn Layout

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from RestraintFile_cs import *    # Class - creates Chemical shift restraint file
from RestraintFile_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from RestraintFile_J import *     # Class - creates J-coupling const. restraint file
from RestraintFile_pf import *	  # Class - creates Protection factor restraint file   #GYH
from restraint_cs_H import *	# test (until compute sse) done for work   Yunhui Ge -- 04/2018
from restraint_J import *
from restraint_pf import *
from restraint_cs_Ha import *
from restraint_cs_N import *
from restraint_cs_Ca import *
from restraint_noe import *

##############################################################################
# Code
##############################################################################

class Restraint(restraint_cs_H, restraint_J, restraint_cs_Ha, restraint_cs_N, restraint_cs_Ca, restraint_noe, restraint_pf):

    """A class to store a molecular structure, its complete set of
    experimental NOE, J-coupling, chemical shift and protection factor data, and
    Each Instances of this object"""

    def __init__(self, PDB_filename, free_energy, data = None,
            use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2,
            gamma_max=10.0):
	"""Initialize the class.
        INPUTS
	PDB_filename	A topology file (*.pdb)                 
        free_energy     The (reduced) free energy f = beta*F of this conformation
	data		input data for BICePs (both model and exp)
	use_log_normal_distances	Not sure what's this...
	dloggamma	gamma is in log space
	gamma_min	min value of gamma
	gamma_max	max value of gamma
        """
        self.PDB_filename = PDB_filename
	self.data = data
	self.conf = mdtraj.load_pdb(PDB_filename)
        # Convert the coordinates from nm to Angstrom units
        self.conf.xyz = self.conf.xyz*10.0

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.free_energy = free_energy

# Rob added this back in to eliminate the errors:{{{
        # Store distance restraint info:
        self.distance_restraints = []
        self.distance_equivalency_groups = {}
        self.ambiguous_groups = []  # list of pairs of group indices, e.g.:   [ [[1,2,3],[4,5,6]],   [[7],[8]], ...]
        self.ndistances = 0

        # Store dihedral restraint info
        self.dihedral_restraints = []
        self.dihedral_equivalency_groups = {}
        self.dihedral_ambiguity_groups = {}
        self.ndihedrals = 0

	# Store chemical shift restraint info   #GYH
        self.cs_H_restraints = []
	self.cs_Ha_restraints = []
	self.cs_N_restraints = []
	self.cs_Ca_restraints = []
        self.ncs_H = 0
	self.ncs_Ha = 0
	self.ncs_N = 0
	self.ncs_Ca = 0

	# Store protection factor restraint info	#GYH
	self.protectionfactor_restraints = []
	self.nprotectionfactor = 0
# }}}

        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_distances = use_log_normal_distances

        # Store info about gamma^(-1/6) scaling  parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))


        # Create a KarplusRelation object
        self.karplus = KarplusRelation()

        # variables to store pre-computed SSE and effective degrees of freedom (d.o.f.)
        self.sse_distances = np.array([0.0 for gamma in self.allowed_gamma])
        self.Ndof_distances = 0.0
        self.sse_dihedrals = None
        self.Ndof_dihedrals = 0.0
        self.sse_cs_H = None #GYH
        self.Ndof_cs_H = None  #GYH
        self.sse_cs_Ha = None #GYH
        self.Ndof_cs_Ha = None  #GYH
        self.sse_cs_N = None #GYH
        self.Ndof_cs_N = None  #GYH
        self.sse_cs_Ca = None #GYH
        self.Ndof_cs_Ca = None  #GYH
	self.sse_protectionfactor = 0.0
	self.Ndof_protectionfactor = 0.0 #GYH
        self.betas_noe = None   # if reference is used, an array of N_j betas for each distance
	self.betas_H = None
	self.betas_Ha = None
	self.betas_N = None
	self.betas_Ca = None
	self.betas_PF = None
        self.neglog_reference_potentials_noe = None
        self.neglog_reference_potentials_H = None
        self.neglog_reference_potentials_Ha = None
        self.neglog_reference_potentials_N = None
        self.neglog_reference_potentials_Ca = None
        self.neglog_reference_potentials_PF = None

	self.ref_sigma_noe = None
	self.ref_mean_noe = None
        self.ref_sigma_H = None
        self.ref_mean_H = None
        self.ref_sigma_Ha = None
        self.ref_mean_Ha = None
        self.ref_sigma_N = None
        self.ref_mean_N = None
        self.ref_sigma_Ca = None
        self.ref_mean_Ca = None
        self.ref_sigma_PF = None
        self.ref_mean_PF = None
        self.gaussian_neglog_reference_potentials_noe = None
        self.gaussian_neglog_reference_potentials_H = None
        self.gaussian_neglog_reference_potentials_Ha = None
        self.gaussian_neglog_reference_potentials_N = None
        self.gaussian_neglog_reference_potentials_Ca = None
        self.gaussian_neglog_reference_potentials_PF = None

        self.sum_neglog_reference_potentials_noe = 0.0	#GYH
        self.sum_neglog_reference_potentials_H = 0.0	#GYH
        self.sum_neglog_reference_potentials_Ha = 0.0	#GYH
        self.sum_neglog_reference_potentials_N = 0.0	#GYH
        self.sum_neglog_reference_potentials_Ca = 0.0	#GYH
        self.sum_neglog_reference_potentials_PF = 0.0	#GYH

        self.sum_gaussian_neglog_reference_potentials_noe = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_H = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_Ha = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_N = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_Ca = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_PF = 0.0      #GYH

        # If an experimental data file is given, load in the information
	r_cs_H = restraint_cs_H()
        r_cs_Ha = restraint_cs_Ha()
        r_cs_N = restraint_cs_N()
        r_cs_Ca = restraint_cs_Ca()
	r_J = restraint_J()
	r_noe = restraint_noe()
	r_pf = restraint_pf()
	if data != None:
		for i in data:
			if i.endswith('.noe'):
				r_noe.load_data_noe(i)
                                self.sse_distances = r_noe.sse_distances
                                self.Ndof_distances = r_noe.Ndof_distances
                                self.distance_restraints = r_noe.distance_restraints
                        elif i.endswith('.J'):
				r_J.load_data_J(i)
                                self.sse_dihedrals = r_J.sse_dihedrals
                                self.Ndof_dihedrals = r_J.Ndof_dihedrals
                                self.dihedral_restraints = r_J.dihedral_restraints
                        elif i.endswith('.cs_H'):
                                r_cs_H.load_data_cs_H(i)
                                self.sse_cs_H = r_cs_H.sse_cs_H
                                self.Ndof_cs_H = r_cs_H.Ndof_cs_H
                                self.cs_H_restraints = r_cs_H.cs_H_restraints
                                self.ncs_H = r_cs_H.ncs_H
                        elif i.endswith('.cs_Ha'):
                                r_cs_Ha.load_data_cs_Ha(i)
                                self.sse_cs_Ha = r_cs_Ha.sse_cs_Ha
                                self.Ndof_cs_Ha = r_cs_Ha.Ndof_cs_Ha
                                self.cs_Ha_restraints = r_cs_Ha.cs_Ha_restraints
                                self.ncs_Ha = r_cs_Ha.ncs_Ha
                        elif i.endswith('.cs_N'):
                                r_cs_N.load_data_cs_N(i)
                                self.sse_cs_N = r_cs_N.sse_cs_N
                                self.Ndof_cs_N = r_cs_N.Ndof_cs_N
                                self.cs_N_restraints = r_cs_N.cs_N_restraints
                        elif i.endswith('.cs_Ca'):
                                r_cs_Ca.load_data_cs_Ca(i)
                                self.sse_cs_Ca = r_cs_Ca.sse_cs_Ca
                                self.Ndof_cs_Ca = r_cs_Ca.Ndof_cs_Ca
                                self.cs_Ca_restraints = r_cs_Ca.cs_Ca_restraints
                        elif i.endswith('.pf'):
                                r_cs_Ca.load_data_pf(i)
                                self.sse_pf = r_pf.sse_pf
                                self.Ndof_pf = r_pf.Ndof_pf
                                self.pf_restraints = r_pf.pf_restraints


                        else:
                            raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
        else:
		raise ValueError("Something is wrong in your input file (necessary input file missing)")

        print "self.sse_cs_H", self.sse_cs_H
        print "self.Ndof_cs_H", self.Ndof_cs_H
# Load Experimental Data (ALL Restraints):{{{


    # Save Experimental Data: ### yaml ### {{{
    def save_expdata(self, filename):

        fout = file(filename, 'w')
        yaml.dump(data, fout)
    # }}}


    # Compute -log( reference potentials (ALL Restraints) ):{{{
    def compute_neglog_reference_potentials_noe(self):		#GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_noe = np.zeros(self.ndistances)
        self.sum_neglog_reference_potentials_noe = 0.
        for j in range(self.ndistances):
            self.neglog_reference_potentials_noe[j] = np.log(self.betas_noe[j]) + self.distance_restraints[j].model_distance/self.betas_noe[j]
            self.sum_neglog_reference_potentials_noe  += self.distance_restraints[j].weight * self.neglog_reference_potentials_noe[j]

    def compute_gaussian_neglog_reference_potentials_noe(self):	#GYH
	"""An alternative option for reference potential based on Gaussian distribution"""
	self.gaussian_neglog_reference_potentials_noe = np.zeros(self.ndistances)
	self.sum_gaussian_neglog_reference_potentials_noe = 0.
	for j in range(self.ndistances):
	    self.gaussian_neglog_reference_potentials_noe[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_noe[j]) + (self.distance_restraints[j].model_distance - self.ref_mean_noe[j])**2.0/(2*(self.ref_sigma_noe[j]**2.0))
	    self.sum_gaussian_neglog_reference_potentials_noe += self.distance_restraints[j].weight * self.gaussian_neglog_reference_potentials_noe[j]

    def compute_neglog_reference_potentials_H(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_H = np.zeros(self.ncs_H)
        self.sum_neglog_reference_potentials_H = 0.
        for j in range(self.ncs_H):
            self.neglog_reference_potentials_H[j] = np.log(self.betas_H[j]) + self.cs_H_restraints[j].model_cs_H/self.betas_H[j]
            self.sum_neglog_reference_potentials_H  += self.cs_H_restraints[j].weight * self.neglog_reference_potentials_H[j]
            print "self.sum_neglog_reference_potentials_H", self.sum_neglog_reference_potentials_H
    def compute_gaussian_neglog_reference_potentials_H(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_H = np.zeros(self.ncs_H)
        self.sum_gaussian_neglog_reference_potentials_H = 0.
        for j in range(self.ncs_H):
            self.gaussian_neglog_reference_potentials_H[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_H[j]) + (self.cs_H_restraints[j].model_cs_H - self.ref_mean_H[j])**2.0/(2*self.ref_sigma_H[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_H += self.cs_H_restraints[j].weight * self.gaussian_neglog_reference_potentials_H[j]


    def compute_neglog_reference_potentials_Ha(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_Ha = np.zeros(self.ncs_Ha)
        self.sum_neglog_reference_potentials_Ha = 0.
        for j in range(self.ncs_Ha):
            self.neglog_reference_potentials_Ha[j] = np.log(self.betas_Ha[j]) + self.cs_Ha_restraints[j].model_cs_Ha/self.betas_Ha[j]
            self.sum_neglog_reference_potentials_Ha  += self.cs_Ha_restraints[j].weight * self.neglog_reference_potentials_Ha[j]

    def compute_gaussian_neglog_reference_potentials_Ha(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_Ha = np.zeros(self.ncs_Ha)
        self.sum_gaussian_neglog_reference_potentials_Ha = 0.
        for j in range(self.ncs_Ha):
            self.gaussian_neglog_reference_potentials_Ha[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ha[j]) + (self.cs_Ha_restraints[j].model_cs_Ha - self.ref_mean_Ha[j])**2.0/(2*self.ref_sigma_Ha[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_Ha += self.cs_Ha_restraints[j].weight * self.gaussian_neglog_reference_potentials_Ha[j]

    def compute_neglog_reference_potentials_N(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_N = np.zeros(self.ncs_N)
        self.sum_neglog_reference_potentials_N = 0.
        for j in range(self.ncs_N):
            self.neglog_reference_potentials_N[j] = np.log(self.betas_N[j]) + self.cs_N_restraints[j].model_cs_N/self.betas_N[j]
            self.sum_neglog_reference_potentials_N  += self.cs_N_restraints[j].weight * self.neglog_reference_potentials_N[j]

    def compute_gaussian_neglog_reference_potentials_N(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_N = np.zeros(self.ncs_N)
        self.sum_gaussian_neglog_reference_potentials_N = 0.
        for j in range(self.ncs_N):
            self.gaussian_neglog_reference_potentials_N[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_N[j]) + (self.cs_N_restraints[j].model_cs_N - self.ref_mean_N[j])**2.0/(2*self.ref_sigma_N[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_N += self.cs_N_restraints[j].weight * self.gaussian_neglog_reference_potentials_N[j]


    def compute_neglog_reference_potentials_Ca(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_potentials_Ca = np.zeros(self.ncs_Ca)
        self.sum_neglog_reference_potentials_Ca = 0.
        for j in range(self.ncs_Ca):
            self.neglog_reference_potentials_Ca[j] = np.log(self.betas_Ca[j]) + self.cs_Ca_restraints[j].model_cs_Ca/self.betas_Ca[j]
            self.sum_neglog_reference_potentials_Ca  += self.cs_Ca_restraints[j].weight * self.neglog_reference_potentials_Ca[j]

    def compute_gaussian_neglog_reference_potentials_Ca(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_Ca = np.zeros(self.ncs_Ca)
        self.sum_gaussian_neglog_reference_potentials_Ca = 0.
        for j in range(self.ncs_Ca):
            self.gaussian_neglog_reference_potentials_Ca[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ca[j]) + (self.cs_Ca_restraints[j].model_cs_Ca - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_Ca += self.cs_Ca_restraints[j].weight * self.gaussian_neglog_reference_potentials_Ca[j]


    def compute_neglog_reference_potentials_PF(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""


        self.neglog_reference_potentials_PF= np.zeros(self.nprotectionfactor)
        self.sum_neglog_reference_potentials_PF = 0.
        for j in range(self.nprotectionfactor):
            self.neglog_reference_potentials_PF[j] = np.log(self.betas_PF[j]) + self.protectionfactor_restraints[j].model_protectionfactor/self.betas_PF[j]
            self.sum_neglog_reference_potentials_PF  += self.protectionfactor_restraints[j].weight * self.neglog_reference_potentials_PF[j]


    def compute_gaussian_neglog_reference_potentials_PF(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_PF = np.zeros(self.nprotectionfactor)
        self.sum_gaussian_neglog_reference_potentials_PF = 0.
        for j in range(self.nprotectionfactor):
            self.gaussian_neglog_reference_potentials_PF[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_PF[j]) + (self.protectionfactor_restraints[j].model_protectionfactor - self.ref_mean_PF[j])**2.0/(2*self.ref_sigma_PF[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_PF += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_potentials_PF[j]
    # }}}

    # Switch Distances and recompute SSE:{{{
    def switch_distances(self, indices1, indices2):
        """Given two lists of ambiguous distance pair indices, switch their distances and recompute the sum of squared errors (SSE)."""
        distance1 = self.distance_restraints[indices1[0]].exp_distance
        distance2 = self.distance_restraints[indices2[0]].exp_distance
        for i in indices1:
            self.distance_restraints[i].exp_distance = distance2
        for j in indices2:
            self.distance_restraints[j].exp_distance = distance1
        self.compute_sse_distances()
    # }}}

    # Compute Dihedral Between 4 Positions:{{{
    def dihedral_angle(self, x0, x1, x2, x3):
        """Calculate the signed dihedral angle between 4 positions.  Result is in degrees."""
        #Calculate Bond Vectors b1, b2, b3
        b1=x1-x0
        b2=x2-x1
        b3=x3-x2

        #Calculate Normal Vectors c1,c2.  This numbering scheme is idiotic, so care.
        c1=np.cross(b2,b3)
        c2=np.cross(b1,b2)

        Arg1=np.dot(b1,c1)
        Arg1*=np.linalg.norm(b2)
        Arg2=np.dot(c2,c1)
        phi=np.arctan2(Arg1,Arg2)

        # return the angle in degrees
        phi*=180./np.pi
        return(phi)

    # }}}

# }}}

