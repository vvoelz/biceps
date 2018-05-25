#!/usr/bin/env python

# Import Modules:{{{
import os, sys, glob
import numpy as np
#from msmbuilder import Conformation
import mdtraj
# Can we get rid of yaml and substitute for another multicolumn layout?
# Ideas:{{{

# }}}

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from RestraintFile_cs import *    # Class - creates Chemical shift restraint file
from RestraintFile_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from RestraintFile_J import *     # Class - creates J-coupling const. restraint file
from RestraintFile_pf import *	  # Class - creates Protection factor restraint file   #GYH

# }}}

# Class Restraint:{{{
class Restraint(object):
    #Notes:# {{{
    '''

    '''
    # }}}

    """A class for all restraints"""

    # __init__:{{{
    def __init__(self, PDB_filename, free_energy, expdata_filename_noe=None,
            expdata_filename_J=None, expdata_filename_cs_H=None,
            expdata_filename_cs_Ha=None, expdata_filename_cs_N=None,
            expdata_filename_cs_Ca=None, expdata_filename_PF=None,
            use_log_normal_distances=False, dloggamma=np.log(1.01),
            gamma_min=0.2, gamma_max=10.0, dalpha=0.1,
            alpha_min=-10.0, alpha_max=10.0):

        """Initialize the class.
        INPUTS
	conf	A molecular structure as an msmbuilder Conformation() object.
                NOTE: For cases where the structure is an ensemble (say, from clustering)
                and the modeled NOE distances and coupling constants are averaged,
                the structure itself can just be a placeholder with the right atom name
                and numbering

        free_energy     The (reduced) free energy f = beta*F of this conformation
        """

        self.PDB_filename = PDB_filename
    	self.expdata_filename_noe = expdata_filename_noe
        self.expdata_filename_J = expdata_filename_J
        self.expdata_filename_cs_H = expdata_filename_cs_H
        self.expdata_filename_cs_Ha = expdata_filename_cs_Ha
        self.expdata_filename_cs_N = expdata_filename_cs_N
        self.expdata_filename_cs_Ca = expdata_filename_cs_Ca
	self.expdata_filename_PF = expdata_filename_PF #GYH
	self.conf = mdtraj.load_pdb(PDB_filename)
        # Convert the coordinates from nm to Angstrom units
        self.conf.xyz = self.conf.xyz*10.0

        # The (reduced) free energy f = beta*F of this structure, as predicted by modeling
        self.free_energy = free_energy

        # Flag to use log-normal distance errors log(d/d0)
        self.use_log_normal_distances = use_log_normal_distances

        # Store info about gamma^(-1/6) scaling  parameter array
        self.dloggamma = dloggamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))

        # Store distance restraint info
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
#	self.sse_protectionfactor = None
	self.sse_protectionfactor = np.array([0.0 for alpha in self.allowed_alpha])  #GYH
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
	self.expdata_filename_noe = expdata_filename_noe
        if expdata_filename_noe != None:
                self.load_expdata_noe(expdata_filename_noe)
        self.expdata_filename_J = expdata_filename_J
        if expdata_filename_J != None:
                self.load_expdata_J(expdata_filename_J)
	self.expdata_filename_cs_H = expdata_filename_cs_H
	if expdata_filename_cs_H != None:
		self.load_expdata_cs_H(expdata_filename_cs_H)	#GYH
        self.expdata_filename_cs_Ha = expdata_filename_cs_Ha
        if expdata_filename_cs_Ha != None:
                self.load_expdata_cs_Ha(expdata_filename_cs_Ha)       #GYH
        self.expdata_filename_cs_N = expdata_filename_cs_N
        if expdata_filename_cs_N != None:
                self.load_expdata_cs_N(expdata_filename_cs_N)       #GYH
        self.expdata_filename_cs_Ca = expdata_filename_cs_Ca
        if expdata_filename_cs_Ca != None:
                self.load_expdata_cs_Ca(expdata_filename_cs_Ca)       #GYH

	self.expdata_filename_PF = expdata_filename_PF
	if expdata_filename_PF != None:
		self.load_expdata_PF(expdata_filename_PF)
    # }}}

    # Restraints:{{{
    def add_distance_restraint(self, i, j, exp_distance, model_distance=None,
                               equivalency_index=None):
        """Add an NOE NMR_Distance() object to the set"""

        # if the modeled distance is not specified, compute the distance from the conformation
        if model_distance == None:
            ri = self.conf.xyz[0,i,:]
            rj = self.conf.xyz[0,j,:]
            dr = rj-ri
            model_distance = np.dot(dr,dr)**0.5

        self.distance_restraints.append( NMR_Distance(i, j, model_distance, exp_distance,
                                                      equivalency_index=equivalency_index) )
        self.ndistances += 1

    def add_dihedral_restraint(self, i, j, k, l, exp_Jcoupling, model_Jcoupling=None,
                               equivalency_index=None, karplus_key="Karplus_HH"):
        """Add a J-coupling NMR_Dihedral() object to the list."""

        # if the modeled Jcoupling value is not specified, compute it from the
        # angle corresponding to the conformation, and the Karplus relation
        if model_Jcoupling == None:
            ri, rj, rk, rl = [self.conf.xyz[0,x,:] for x in [i, j, k, l]]
            model_angle = self.dihedral_angle(ri,rj,rk,rl)

            ###########################
            # NOTE: In the future, this function can be more sophisticated, parsing atom types
            # or karplus relation types on a case-by-case basis
            model_Jcoupling = self.karplus.J(model_angle, karplus_key)

        self.dihedral_restraints.append( NMR_Dihedral(i, j, k, l, model_Jcoupling, exp_Jcoupling, model_angle,
                                                      equivalency_index=equivalency_index) )
        self.ndihedrals += 1

    def add_chemicalshift_H_restraint(self, i, exp_chemicalshift_H, model_chemicalshift_H=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
        if model_chemicalshift_H == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pH=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
                model_chemicalshift_H = 1  # will be replaced by pre-computed cs

        self.chemicalshift_H_restraints.append( NMR_Chemicalshift_H(i, model_chemicalshift_H, exp_chemicalshift_H))

        self.nchemicalshift_H += 1

    def add_chemicalshift_Ha_restraint(self, i, exp_chemicalshift_Ha, model_chemicalshift_Ha=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
        if model_chemicalshift_Ha == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pH=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
                model_chemicalshift_Ha = 1  # will be replaced by pre-computed cs
        self.chemicalshift_Ha_restraints.append( NMR_Chemicalshift_Ha(i, model_chemicalshift_Ha, exp_chemicalshift_Ha))

        self.nchemicalshift_Ha += 1

    def add_chemicalshift_Ca_restraint(self, i, exp_chemicalshift_Ca, model_chemicalshift_Ca=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
        if model_chemicalshift_Ca == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pH=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
                model_chemicalshift_Ca = 1  # will be replaced by pre-computed cs
        self.chemicalshift_Ca_restraints.append( NMR_Chemicalshift_Ca(i, model_chemicalshift_Ca, exp_chemicalshift_Ca))

        self.nchemicalshift_Ca += 1

    def add_chemicalshift_N_restraint(self, i, exp_chemicalshift_N, model_chemicalshift_N=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
        if model_chemicalshift_N == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pH=2.5, temperature = 280.0)

 #              model_chemicalshift = r.mean(axis=1)
                model_chemicalshift_N = 1  # will be replaced by pre-computed cs
        self.chemicalshift_N_restraints.append( NMR_Chemicalshift_N(i, model_chemicalshift_N, exp_chemicalshift_N))

        self.nchemicalshift_N += 1



    def add_protectionfactor_restraint(self, i, exp_protectionfactor, model_protectionfactor=None):
	"""Add a protectionfactor NMR_Protectionfactor() object to the list."""
	if model_protectionfactor == None:
		model_protectionfactor = 1 #GYH: will be replaced by pre-computed PF
	self.protectionfactor_restraints.append(NMR_Protectionfactor(i, model_protectionfactor, exp_protectionfactor))
	self.nprotectionfactor += 1
    # }}}




#}}}





