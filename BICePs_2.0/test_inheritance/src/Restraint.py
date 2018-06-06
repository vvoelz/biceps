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
from prep_cs import *    # Class - creates Chemical shift restraint file
from prep_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from prep_J import *     # Class - creates J-coupling const. restraint file
from prep_pf import *	  # Class - creates Protection factor restraint file   #GYH
#from restraint_cs_H import *	# test (until compute sse) done for work   Yunhui Ge -- 04/2018
#from restraint_J import *
#from restraint_pf import *
#from restraint_cs_Ha import *
#from restraint_cs_N import *
#from restraint_cs_Ca import *
#from restraint_noe import *
from inheri import *
#import inheri
from toolbox import *
##############################################################################
# Code
##############################################################################

#class Restraint(restraint_cs_H, restraint_J, restraint_cs_Ha, restraint_cs_N, restraint_cs_Ca, restraint_noe, restraint_pf):
#class Restraint(restraint_cs_H):
class Restraint(inheri):
    """A class to store a molecular structure, its complete set of
    experimental NOE, J-coupling, chemical shift and protection factor data, and
    Each Instances of this object"""

    def __init__(self, PDB_filename, lam, free_energy, data = None,
            use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2,
            gamma_max=10.0):
	"""Initialize the class.
        INPUTS
	PDB_filename	A topology file (*.pdb)
	lam		lambda value (between 0 and 1)                 
        free_energy     The (reduced) free energy f = beta*F of this conformation
	data		input data for BICePs (both model and exp)
	use_log_normal_distances	Not sure what's this...
	dloggamma	gamma is in log space
	gamma_min	min value of gamma
	gamma_max	max value of gamma
        """
	inheri.__init__(self)

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
        self.use_log_normal_distances = use_log_normal_distances

	# initialize all restraint child class
	restraint_noe(gamma_min=self.gamma_min,gamma_max=self.gamma_max,dloggamma=self.dloggamma,use_log_normal_distances=self.use_log_normal_distances)	
#	r_noe = restraint_noe(gamma_min=self.gamma_min,gamma_max=self.gamma_max,dloggamma=self.dloggamma,use_log_normal_distances=self.use_log_normal_distances)
#        r_cs_H = restraint_cs_H()
#        r_cs_Ha = restraint_cs_Ha()
#        r_cs_N = restraint_cs_N()
#        r_cs_Ca = restraint_cs_Ca()
#        r_J = restraint_J()
#        r_pf = restraint_pf()



        # Create a KarplusRelation object
        self.karplus = KarplusRelation()

        # variables to store pre-computed SSE and effective degrees of freedom (d.o.f.)
        self.betas_noe = None   # if reference is used, an array of N_j betas for each distance
	self.betas_H = None
	self.betas_Ha = None
	self.betas_N = None
	self.betas_Ca = None
	self.betas_pf = None
        self.neglog_reference_potentials_noe = None
        self.neglog_reference_potentials_H = None
        self.neglog_reference_potentials_Ha = None
        self.neglog_reference_potentials_N = None
        self.neglog_reference_potentials_Ca = None
        self.neglog_reference_potentials_pf = None

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
        self.ref_sigma_pf = None
        self.ref_mean_pf = None
        self.gaussian_neglog_reference_potentials_noe = None
        self.gaussian_neglog_reference_potentials_H = None
        self.gaussian_neglog_reference_potentials_Ha = None
        self.gaussian_neglog_reference_potentials_N = None
        self.gaussian_neglog_reference_potentials_Ca = None
        self.gaussian_neglog_reference_potentials_pf = None

        self.sum_neglog_reference_potentials_noe = 0.0	#GYH
        self.sum_neglog_reference_potentials_H = 0.0	#GYH
        self.sum_neglog_reference_potentials_Ha = 0.0	#GYH
        self.sum_neglog_reference_potentials_N = 0.0	#GYH
        self.sum_neglog_reference_potentials_Ca = 0.0	#GYH
        self.sum_neglog_reference_potentials_pf = 0.0	#GYH

        self.sum_gaussian_neglog_reference_potentials_noe = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_H = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_Ha = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_N = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_Ca = 0.0      #GYH
        self.sum_gaussian_neglog_reference_potentials_pf = 0.0      #GYH

        # If an experimental data file is given, load in the information
	if data != None:
		for i in data:
#			print i
			if i.endswith('.noe'):
#				restraint_noe(gamma_min=self.gamma_min,gamma_max=self.gamma_max,dloggamma=self.dloggamma,use_log_normal_distances=self.use_log_normal_distances)
				self.load_data_noe(i)
                        elif i.endswith('.J'):
				self.load_data_J(i)
                        elif i.endswith('.cs_H'):
                   #             r_cs_H.load_data_cs_H(i)
                   		self.load_data_cs_H(i)
		        elif i.endswith('.cs_Ha'):
                                self.load_data_cs_Ha(i)
                        elif i.endswith('.cs_N'):
                                self.load_data_cs_N(i)
                        elif i.endswith('.cs_Ca'):
                                self.load_data_cs_Ca(i)
                        elif i.endswith('.pf'):
                                self.load_data_pf(i)


                        else:
                            raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha, .cs_Ca, .cs_N,.pf}")
        else:
		raise ValueError("Something is wrong in your input file (necessary input file missing)")
	print "self.sse_distances", self.sse_distances

#        self.sse_distances = r_noe.sse_distances
#        self.Ndof_distances = r_noe.Ndof_distances
#        self.distance_restraints = r_noe.distance_restraints
#        self.distance_equivalency_groups = r_noe.distance_equivalency_groups
#        self.ndistances = r_noe.ndistances
#        self.sse_dihedrals = r_J.sse_dihedrals
#        self.Ndof_dihedrals = r_J.Ndof_dihedrals
#        self.dihedral_restraints = r_J.dihedral_restraints
#        self.dihedral_equivalency_groups = r_J.dihedral_equivalency_groups
#        self.ndihedrals = r_J.ndihedrals
#        self.sse_cs_H = r_cs_H.sse_cs_H
#        self.Ndof_cs_H = r_cs_H.Ndof_cs_H
#        self.cs_H_restraints = r_cs_H.cs_H_restraints
#        self.ncs_H = r_cs_H.ncs_H
#        self.sse_cs_Ha = r_cs_Ha.sse_cs_Ha
#        self.Ndof_cs_Ha = r_cs_Ha.Ndof_cs_Ha
#        self.cs_Ha_restraints = r_cs_Ha.cs_Ha_restraints
#        self.ncs_Ha = r_cs_Ha.ncs_Ha
#        self.sse_cs_N = r_cs_N.sse_cs_N
#        self.Ndof_cs_N = r_cs_N.Ndof_cs_N
#        self.cs_N_restraints = r_cs_N.cs_N_restraints
#        self.ncs_N = r_cs_N.ncs_N
#        self.sse_cs_Ca = r_cs_Ca.sse_cs_Ca
#        self.Ndof_cs_Ca = r_cs_Ca.Ndof_cs_Ca
#        self.cs_Ca_restraints = r_cs_Ca.cs_Ca_restraints
#        self.ncs_Ca = r_cs_Ca.ncs_Ca
#        self.sse_pf = r_pf.sse_pf
#        self.Ndof_pf = r_pf.Ndof_pf
#        self.pf_restraints = r_pf.pf_restraints
#        self.npf = r_pf.npf



#        print "self.sse_cs_H", self.sse_cs_H
#        print "self.Ndof_cs_H", self.Ndof_cs_H
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


    def compute_neglog_reference_potentials_pf(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""


        self.neglog_reference_potentials_pf= np.zeros(self.nprotectionfactor)
        self.sum_neglog_reference_potentials_pf = 0.
        for j in range(self.nprotectionfactor):
            self.neglog_reference_potentials_pf[j] = np.log(self.betas_pf[j]) + self.protectionfactor_restraints[j].model_protectionfactor/self.betas_pf[j]
            self.sum_neglog_reference_potentials_pf  += self.protectionfactor_restraints[j].weight * self.neglog_reference_potentials_pf[j]


    def compute_gaussian_neglog_reference_potentials_pf(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_potentials_pf = np.zeros(self.nprotectionfactor)
        self.sum_gaussian_neglog_reference_potentials_pf = 0.
        for j in range(self.nprotectionfactor):
            self.gaussian_neglog_reference_potentials_pf[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_pf[j]) + (self.protectionfactor_restraints[j].model_protectionfactor - self.ref_mean_pf[j])**2.0/(2*self.ref_sigma_pf[j]**2.0)
            self.sum_gaussian_neglog_reference_potentials_pf += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_potentials_pf[j]


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

