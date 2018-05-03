#!/usr/bin/env python

# Import Modules:{{{
import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj
import yaml                       # Yet Another Multicolumn Layout
# Can we get rid of yaml and substitute for another multicolumn layout?
# Ideas:{{{

# }}}

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from RestraintFile_cs import *    # Class - creates Chemical shift restraint file
from RestraintFile_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from RestraintFile_J import *     # Class - creates J-coupling const. restraint file
from RestraintFile_pf import *	  # Class - creates Protection factor restraint file   #GYH
from restraint_cs_H import *	# test (until compute sse) done for work   Yunhui Ge -- 04/2018
from restraint_J import *
from restraint_cs_Ha import *
from restraint_cs_N import *
from restraint_cs_Ca import *
from restraint_noe import *
# }}}

# Class Structure:{{{
class Structure(restraint_cs_H, restraint_J, restraint_cs_Ha, restraint_cs_N, restraint_cs_Ca, restraint_noe):
    #Notes:# {{{
    '''
     ambiguous groups are not yet supported
    '''
    # }}}

    """A class to store a molecular structure, its complete set of
    experimental NOE, J-coupling and chemical shift data, and
    Each Instances of this obect"""

    # __init__:{{{
#    def __init__(self, PDB_filename, free_energy, expdata_filename_noe=None, expdata_filename_J=None, expdata_filename_cs_H=None, expdata_filename_cs_Ha=None, expdata_filename_cs_N=None, expdata_filename_cs_Ca=None, expdata_filename_PF=None, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=10.0, dalpha=0.1, alpha_min=-10.0, alpha_max=10.0):
    def __init__(self, PDB_filename, free_energy, data = None,
            use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2,
            gamma_max=10.0, dalpha=0.1, alpha_min=-10.0,alpha_max=10.0):
	"""Initialize the class.
        INPUTS
	conf		A molecular structure as an msmbuilder Conformation() object.
                        NOTE: For cases where the structure is an ensemble (say, from clustering)
                        and the modeled NOE distances and coupling constants are averaged,
                        the structure itself can just be a placeholder with the right atom name
                        and numbering
        free_energy     The (reduced) free energy f = beta*F of this conformation
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
        self.chemicalshift_H_restraints = []
	self.chemicalshift_Ha_restraints = []
	self.chemicalshift_N_restraints = []
	self.chemicalshift_Ca_restraints = []
#        self.chemicalshift_equivalency_groups = {}
#        self.chemicalshift_ambiguity_groups = {}
        self.nchemicalshift_H = 0
	self.nchemicalshift_Ha = 0
	self.nchemicalshift_N = 0
	self.nchemicalshift_Ca = 0

	# Store protection factor restraint info	#GYH
	self.protectionfactor_restraints = []
#	self.protectionfactor_equivalency_groups = {}
#	self.protectionfactor_ambiguity_groups = {}
	self.nprotectionfactor = 0
# }}}

        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_distances = use_log_normal_distances

        # Store info about gamma^(-1/6) scaling  parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))

        # Store info about alpha 'scaling'  parameter array #GYH
        self.dalpha = dalpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.allowed_alpha = np.arange(self.alpha_min, self.alpha_max, self.dalpha)


        # Create a KarplusRelation object
        self.karplus = KarplusRelation()

        # variables to store pre-computed SSE and effective degrees of freedom (d.o.f.)
        self.sse_distances = np.array([0.0 for gamma in self.allowed_gamma])
        self.Ndof_distances = 0.0
        self.sse_dihedrals = None
        self.Ndof_dihedrals = 0.0
        self.sse_chemicalshift_H = None #GYH
        self.Ndof_chemicalshift_H = None  #GYH
        self.sse_chemicalshift_Ha = None #GYH
        self.Ndof_chemicalshift_Ha = None  #GYH
        self.sse_chemicalshift_N = None #GYH
        self.Ndof_chemicalshift_N = None  #GYH
        self.sse_chemicalshift_Ca = None #GYH
        self.Ndof_chemicalshift_Ca = None  #GYH
	self.sse_protectionfactor = None
#	self.sse_protectionfactor = np.array([0.0 for alpha in self.allowed_alpha])  #GYH
	self.Ndof_protectionfactor = None #GYH
        self.betas_noe = None   # if reference is used, an array of N_j betas for each distance
	self.betas_H = None
	self.betas_Ha = None
	self.betas_N = None
	self.betas_Ca = None
	self.betas_PF = None
        self.neglog_reference_priors_noe = None
        self.neglog_reference_priors_H = None
        self.neglog_reference_priors_Ha = None
        self.neglog_reference_priors_N = None
        self.neglog_reference_priors_Ca = None
        self.neglog_reference_priors_PF = None

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
        self.gaussian_neglog_reference_priors_noe = None
        self.gaussian_neglog_reference_priors_H = None
        self.gaussian_neglog_reference_priors_Ha = None
        self.gaussian_neglog_reference_priors_N = None
        self.gaussian_neglog_reference_priors_Ca = None
        self.gaussian_neglog_reference_priors_PF = None

        self.sum_neglog_reference_priors_noe = 0.0	#GYH
        self.sum_neglog_reference_priors_H = 0.0	#GYH
        self.sum_neglog_reference_priors_Ha = 0.0	#GYH
        self.sum_neglog_reference_priors_N = 0.0	#GYH
        self.sum_neglog_reference_priors_Ca = 0.0	#GYH
        self.sum_neglog_reference_priors_PF = 0.0	#GYH

        self.sum_gaussian_neglog_reference_priors_noe = 0.0      #GYH
        self.sum_gaussian_neglog_reference_priors_H = 0.0      #GYH
        self.sum_gaussian_neglog_reference_priors_Ha = 0.0      #GYH
        self.sum_gaussian_neglog_reference_priors_N = 0.0      #GYH
        self.sum_gaussian_neglog_reference_priors_Ca = 0.0      #GYH
        self.sum_gaussian_neglog_reference_priors_PF = 0.0      #GYH

        # If an experimental data file is given, load in the information
	r_cs_H = restraint_cs_H()
        r_cs_Ha = restraint_cs_Ha()
        r_cs_N = restraint_cs_N()
        r_cs_Ca = restraint_cs_Ca()
	r_J = restraint_J()
	r_noe = restraint_noe()
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
                                self.sse_chemicalshift_H = r_cs_H.sse_chemicalshift_H
                                self.Ndof_chemicalshift_H = r_cs_H.Ndof_chemicalshift_H
                                self.chemicalshift_H_restraints = r_cs_H.chemicalshift_H_restraints
                                self.nchemicalshift_H = r_cs_H.nchemicalshift_H
                        elif i.endswith('.cs_Ha'):
                                r_cs_Ha.load_data_cs_Ha(i)
                                self.sse_chemicalshift_Ha = r_cs_Ha.sse_chemicalshift_Ha
                                self.Ndof_chemicalshift_Ha = r_cs_Ha.Ndof_chemicalshift_Ha
                                self.chemicalshift_Ha_restraints = r_cs_Ha.chemicalshift_Ha_restraints
                                self.nchemicalshift_Ha = r_cs_Ha.nchemicalshift_Ha
                        elif i.endswith('.cs_N'):
                                r_cs_N.load_data_cs_N(i)
                                self.sse_chemicalshift_N = r_cs_N.sse_chemicalshift_N
                                self.Ndof_chemicalshift_N = r_cs_N.Ndof_chemicalshift_N
                                self.chemicalshift_N_restraints = r_cs_N.chemicalshift_N_restraints
                        elif i.endswith('.cs_Ca'):
                                r_cs_Ca.load_data_cs_Ca(i)
                                self.sse_chemicalshift_Ca = r_cs_Ca.sse_chemicalshift_Ca
                                self.Ndof_chemicalshift_Ca = r_cs_Ca.Ndof_chemicalshift_Ca
                                self.chemicalshift_Ca_restraints = r_cs_Ca.chemicalshift_Ca_restraints
                        else:
                            raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
        else:
		raise ValueError("Something is wrong in your input file (necessary input file missing)")

        print "self.sse_chemicalshift_H", self.sse_chemicalshift_H
        print "self.Ndof_chemicalshift_H", self.Ndof_chemicalshift_H
#        print "self.chemicalshift_H_restraints", self.chemicalshift_H_restraints
# Load Experimental Data (ALL Restraints):{{{

    def load_expdata_PF(self, filename, verbose=False):
        """Load in the experimental protection factor restraints from a .PF file format.
        """

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_pf(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_pf(line) )  # [restraint_index, atom_index1, res1, atom_name1, protectionfactor] #GYH: need adjust once data are available!!!

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for protection factors are in the first column of the *.PF file
    #    equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'protectionfactor equivalency_indices', equivalency_indices


        # add the protection factor restraints
        for entry in data:				#GYH
            restraint_index, i, exp_protectionfactor = entry[0], entry[1], entry[3]
            self.add_protectionfactor_restraint(i, exp_protectionfactor, model_protectionfactor=None)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_pf()
    # }}}

    # Save Experimental Data: ### yaml ### {{{
    def save_expdata(self, filename):

        fout = file(filename, 'w')
        yaml.dump(data, fout)
    # }}}

    # Restraints:{{{


    def add_protectionfactor_restraint(self, i, exp_protectionfactor, model_protectionfactor=None):
	"""Add a protectionfactor NMR_Protectionfactor() object to the list."""
	if model_protectionfactor == None:
		model_protectionfactor = 1 #GYH: will be replaced by pre-computed PF
	self.protectionfactor_restraints.append(NMR_Protectionfactor(i, model_protectionfactor, exp_protectionfactor))
	self.nprotectionfactor += 1
    # }}}



    def build_groups_pf(self, verbose=False):
#        """Build equivalency and ambiguity groups for distances and dihedrals,
#        and store pre-computed SSE and d.o.f for distances and dihedrals"""


	# compile protectionfactor_equivalency_groups from the list of NMR_Protectionfactor() objects	#GYH
#	for i in range(len(self.protectionfactor_restraints)):
#	    d = self.protectionfactor_restraints[i]
#	    if d.equivalency_index != None:
#		if not self.protectionfactor_equivalency_groups.has_key(d.equivalency_index):
#		   self.protectionfactor_equivalency_groups[d.equivalency_index] = []
#		self.protectionfactor_equivalency_groups[d.equivalency_index].append(i)
#	if verbose:
#	    print 'self.protectionfactor_equivalency_groups', self.protectionfactor_equivalency_groups


	# precompute SSE and Ndof for protection factor #GYH
	self.compute_sse_protectionfactor()
    # }}}

    # Compute SSE, Sum Squared Errors (ALL Restraints):{{{

    def compute_sse_protectionfactor(self,debug=False):		#GYH
	"""Returns the (weighted) sum of squared errors for protection factor values"""
	for a in range(len(self.allowed_alpha)):

		sse = 0.0
		N = 0.0
		for i in range(self.nprotectionfactor):
			alpha = self.allowed_alpha[a]
			if a == 0:
				print '---->', i, '(%d)'%(self.protectionfactor_restraints[i].i),
	                        print '      exp',  self.protectionfactor_restraints[i].exp_protectionfactor, 'model', self.protectionfactor_restraints[i].model_protectionfactor

			err=self.protectionfactor_restraints[i].model_protectionfactor - self.protectionfactor_restraints[i].exp_protectionfactor - alpha
			sse += (self.protectionfactor_restraints[i].weight * err**2.0)
			N += self.protectionfactor_restraints[i].weight
		self.sse_protectionfactor[a] = sse
		self.Ndof_protectionfactor = N
	if debug:
	    print 'self.sse_protectionfactor', self.sse_protectionfactor
    # }}}

    # Compute -log( reference priors (ALL Restraints) ):{{{
    def compute_neglog_reference_priors_noe(self):		#GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors_noe = np.zeros(self.ndistances)
        self.sum_neglog_reference_priors_noe = 0.
        for j in range(self.ndistances):
            self.neglog_reference_priors_noe[j] = np.log(self.betas_noe[j]) + self.distance_restraints[j].model_distance/self.betas_noe[j]
            self.sum_neglog_reference_priors_noe  += self.distance_restraints[j].weight * self.neglog_reference_priors_noe[j]

    def compute_gaussian_neglog_reference_priors_noe(self):	#GYH
	"""An alternative option for reference potential based on Gaussian distribution"""
	self.gaussian_neglog_reference_priors_noe = np.zeros(self.ndistances)
	self.sum_gaussian_neglog_reference_priors_noe = 0.
	for j in range(self.ndistances):
	    self.gaussian_neglog_reference_priors_noe[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_noe[j]) + (self.distance_restraints[j].model_distance - self.ref_mean_noe[j])**2.0/(2*(self.ref_sigma_noe[j]**2.0))
	    self.sum_gaussian_neglog_reference_priors_noe += self.distance_restraints[j].weight * self.gaussian_neglog_reference_priors_noe[j]

    def compute_neglog_reference_priors_H(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors_H = np.zeros(self.nchemicalshift_H)
        self.sum_neglog_reference_priors_H = 0.
        for j in range(self.nchemicalshift_H):
            self.neglog_reference_priors_H[j] = np.log(self.betas_H[j]) + self.chemicalshift_H_restraints[j].model_chemicalshift_H/self.betas_H[j]
            self.sum_neglog_reference_priors_H  += self.chemicalshift_H_restraints[j].weight * self.neglog_reference_priors_H[j]
            print "self.sum_neglog_reference_priors_H", self.sum_neglog_reference_priors_H
    def compute_gaussian_neglog_reference_priors_H(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_priors_H = np.zeros(self.nchemicalshift_H)
        self.sum_gaussian_neglog_reference_priors_H = 0.
        for j in range(self.nchemicalshift_H):
            self.gaussian_neglog_reference_priors_H[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_H[j]) + (self.chemicalshift_H_restraints[j].model_chemicalshift_H - self.ref_mean_H[j])**2.0/(2*self.ref_sigma_H[j]**2.0)
            self.sum_gaussian_neglog_reference_priors_H += self.chemicalshift_H_restraints[j].weight * self.gaussian_neglog_reference_priors_H[j]


    def compute_neglog_reference_priors_Ha(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors_Ha = np.zeros(self.nchemicalshift_Ha)
        self.sum_neglog_reference_priors_Ha = 0.
        for j in range(self.nchemicalshift_Ha):
            self.neglog_reference_priors_Ha[j] = np.log(self.betas_Ha[j]) + self.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha/self.betas_Ha[j]
            self.sum_neglog_reference_priors_Ha  += self.chemicalshift_Ha_restraints[j].weight * self.neglog_reference_priors_Ha[j]

    def compute_gaussian_neglog_reference_priors_Ha(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_priors_Ha = np.zeros(self.nchemicalshift_Ha)
        self.sum_gaussian_neglog_reference_priors_Ha = 0.
        for j in range(self.nchemicalshift_Ha):
            self.gaussian_neglog_reference_priors_Ha[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ha[j]) + (self.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha - self.ref_mean_Ha[j])**2.0/(2*self.ref_sigma_Ha[j]**2.0)
            self.sum_gaussian_neglog_reference_priors_Ha += self.chemicalshift_Ha_restraints[j].weight * self.gaussian_neglog_reference_priors_Ha[j]

    def compute_neglog_reference_priors_N(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors_N = np.zeros(self.nchemicalshift_N)
        self.sum_neglog_reference_priors_N = 0.
        for j in range(self.nchemicalshift_N):
            self.neglog_reference_priors_N[j] = np.log(self.betas_N[j]) + self.chemicalshift_N_restraints[j].model_chemicalshift_N/self.betas_N[j]
            self.sum_neglog_reference_priors_N  += self.chemicalshift_N_restraints[j].weight * self.neglog_reference_priors_N[j]

    def compute_gaussian_neglog_reference_priors_N(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_priors_N = np.zeros(self.nchemicalshift_N)
        self.sum_gaussian_neglog_reference_priors_N = 0.
        for j in range(self.nchemicalshift_N):
            self.gaussian_neglog_reference_priors_N[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_N[j]) + (self.chemicalshift_N_restraints[j].model_chemicalshift_N - self.ref_mean_N[j])**2.0/(2*self.ref_sigma_N[j]**2.0)
            self.sum_gaussian_neglog_reference_priors_N += self.chemicalshift_N_restraints[j].weight * self.gaussian_neglog_reference_priors_N[j]


    def compute_neglog_reference_priors_Ca(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors_Ca = np.zeros(self.nchemicalshift_Ca)
        self.sum_neglog_reference_priors_Ca = 0.
        for j in range(self.nchemicalshift_Ca):
            self.neglog_reference_priors_Ca[j] = np.log(self.betas_Ca[j]) + self.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca/self.betas_Ca[j]
            self.sum_neglog_reference_priors_Ca  += self.chemicalshift_Ca_restraints[j].weight * self.neglog_reference_priors_Ca[j]

    def compute_gaussian_neglog_reference_priors_Ca(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_priors_Ca = np.zeros(self.nchemicalshift_Ca)
        self.sum_gaussian_neglog_reference_priors_Ca = 0.
        for j in range(self.nchemicalshift_Ca):
            self.gaussian_neglog_reference_priors_Ca[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_Ca[j]) + (self.chemicalshift_Ca_restraints[j].model_chemicalshift_Ca - self.ref_mean_Ca[j])**2.0/(2*self.ref_sigma_Ca[j]**2.0)
            self.sum_gaussian_neglog_reference_priors_Ca += self.chemicalshift_Ca_restraints[j].weight * self.gaussian_neglog_reference_priors_Ca[j]


    def compute_neglog_reference_priors_PF(self):              #GYH
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors_PF= np.zeros(self.nprotectionfactor)
        self.sum_neglog_reference_priors_PF = 0.
        for j in range(self.nprotectionfactor):
            self.neglog_reference_priors_PF[j] = np.log(self.betas_PF[j]) + self.protectionfactor_restraints[j].model_protectionfactor/self.betas_PF[j]
            self.sum_neglog_reference_priors_PF  += self.protectionfactor_restraints[j].weight * self.neglog_reference_priors_PF[j]


    def compute_gaussian_neglog_reference_priors_PF(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_priors_PF = np.zeros(self.nprotectionfactor)
        self.sum_gaussian_neglog_reference_priors_PF = 0.
        for j in range(self.nprotectionfactor):
#	    print j, 'self.ref_sigma_PF[j]', self.ref_sigma_PF[j], 'self.ref_mean_PF[j]', self.ref_mean_PF[j]
            self.gaussian_neglog_reference_priors_PF[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_PF[j]) + (self.protectionfactor_restraints[j].model_protectionfactor - self.ref_mean_PF[j])**2.0/(2*self.ref_sigma_PF[j]**2.0)
            self.sum_gaussian_neglog_reference_priors_PF += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_priors_PF[j]
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

# Class NMR_Protectionfactor:{{{
class NMR_Protectionfactor(object):        #GYH
    """A class to store NMR protection factor information."""

    # __init__:{{{
    def __init__(self, i, model_protectionfactor, exp_protectionfactor):
        # Atom indices from the Conformation() defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model_protectionfactor = model_protectionfactor

        # the experimental protection factor
        self.exp_protectionfactor = exp_protectionfactor


        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1
    # }}}

# }}}

