import os, sys, glob
import numpy as np

#from msmbuilder import Conformation
import mdtraj
import yaml

from KarplusRelation import *
from RestraintFile_cs import *
from RestraintFile_noe import *
from RestraintFile_J import *
from RestraintFile_pf import *		#GYH


class Structure(object):
    """A class to store a molecular structure, its complete set of
    experimental NOE, J-coupling and chemical shift data, and   
    Each Instances of this obect"""

#    def __init__(self, PDB_filename, free_energy, expdata_filename_noe=None, expdata_filename_J=None, expdata_filename_cs_H=None, expdata_filename_cs_Ha=None, expdata_filename_cs_N=None, expdata_filename_cs_Ca=None, expdata_filename_PF=None, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=10.0, dbeta_c=0.005, beta_c_min=0.02, beta_c_max=0.095, dbeta_h=0.05, beta_h_min=0.00, beta_h_max=2.00, dbeta_0=0.2, beta_0_min=-3.0, beta_0_max=1.0, dxcs=0.5, xcs_min=5.0, xcs_max=8.5, dxhs=0.1, xhs_min=2.0, xhs_max=2.8, dbs=1.0, bs_min=3.0, bs_max=21.0, Ncs=None, Nhs=None):	#GYH 03/2017
    def __init__(self, PDB_filename, free_energy, expdata_filename_noe=None, expdata_filename_J=None, expdata_filename_cs_H=None, expdata_filename_cs_Ha=None, expdata_filename_cs_N=None, expdata_filename_cs_Ca=None, expdata_filename_PF=None, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=10.0, dbeta_c=0.005, beta_c_min=0.02, beta_c_max=0.03, dbeta_h=0.05, beta_h_min=0.00, beta_h_max=0.10, dbeta_0=0.2, beta_0_min=0.0, beta_0_max=0.4, dxcs=0.5, xcs_min=5.0, xcs_max=6.0, dxhs=0.1, xhs_min=2.0, xhs_max=2.1, dbs=1.0, bs_min=3.0, bs_max=5.0, Ncs=None, Nhs=None): #GYH 05/2017
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

        # Store info about PF calculation  parameter array  # GYH 03/2017
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

	# parameters for Gaussian reference potential
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

    def load_expdata_noe(self, filename, verbose=False):
        """Load in the experimental NOE distance restraints from a .noe file format.
	"""

        # Read in the lines of the biceps data file
        b = RestraintFile_noe(filename=filename)
        data = []
        for line in b.lines:
		data.append( b.parse_line_noe(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, distance]
         
        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'distance equivalency_indices', equivalency_indices

        # compile ambiguity indices for distances
        """ ### not yet supported ###
        
          for pair in data['NOE_Ambiguous']:
            # NOTE a pair of multiple distance pairs
            list1, list2 = pair[0], pair[1]
            # find the indices of the distances pairs that are ambiguous
            pair_indices1 = [ data['NOE_PairIndex'].index(p) for p in list1]
            pair_indices2 = [ data['NOE_PairIndex'].index(p) for p in list2]
            self.ambiguous_groups.append( [pair_indices1, pair_indices2] )
          if verbose:
            print 'distance ambiguous_groups', self.ambiguous_groups
        except:
            print 'Problem reading distance ambiguous_groups.  Setting to default: no ambiguous groups.'
        """

        # add the distance restraints
        for entry in data:
            restraint_index, i, j, exp_distance = entry[0], entry[1], entry[4], entry[7]
            self.add_distance_restraint(i, j, exp_distance, model_distance=None, equivalency_index=restraint_index)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_noe()


    def load_expdata_J(self, filename, verbose=False):
        """Load in the experimental Jcoupling constant restraints from a .Jcoupling file format."""


        # Read in the lines of the biceps data file
        b = RestraintFile_J(filename=filename)
        data = []
        for line in b.lines:
                data.append( b.parse_line_J(line) )  # [restraint_index, atom_index1, res1, atom_name1, atom_index2, res2, atom_name2, atom_index3, res3, atom_name3, atom_index4, res4, atom_name4, J_coupling(Hz)]

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
            restraint_index, i, j, k, l, exp_Jcoupling, karplus  = entry[0], entry[1], entry[4], entry[7], entry[10], entry[13], entry[14]
            self.add_dihedral_restraint(i, j, k, l, exp_Jcoupling, model_Jcoupling=None, equivalency_index=None, karplus_key=karplus)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_J()



    def load_expdata_cs_H(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
	"""

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_cs(filename=filename)
        if verbose:
		print b.lines
	data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_H = entry[0], entry[1], entry[4]
            self.add_chemicalshift_H_restraint(i, exp_chemicalshift_H, model_chemicalshift_H=None)

        # build groups of equivalency group indices, ambiguous group etc.

        self.build_groups_cs_H()


    def load_expdata_cs_Ha(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_Ha = entry[0], entry[1], entry[4]
            self.add_chemicalshift_Ha_restraint(i, exp_chemicalshift_Ha, model_chemicalshift_Ha=None)

        # build groups of equivalency group indices, ambiguous group etc.

        self.build_groups_cs_Ha()

    def load_expdata_cs_N(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_N = entry[0], entry[1], entry[4]
            self.add_chemicalshift_N_restraint(i, exp_chemicalshift_N, model_chemicalshift_N=None)

        # build groups of equivalency group indices, ambiguous group etc.

        self.build_groups_cs_N()


    def load_expdata_cs_Ca(self, filename, verbose=False):
        """Load in the experimental chemical shift restraints from a .chemicalshift file format.
        """

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_cs(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_cs(line) )  # [restraint_index, atom_index1, res1, atom_name1, chemicalshift]

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for distances are in the first column of the *.biceps f
#       equivalency_indices = [entry[0] for entry in data]


        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift_Ca = entry[0], entry[1], entry[4]
            self.add_chemicalshift_Ca_restraint(i, exp_chemicalshift_Ca, model_chemicalshift_Ca=None)

        # build groups of equivalency group indices, ambiguous group etc.

        self.build_groups_cs_Ca()





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
#            restraint_index, i, exp_protectionfactor = entry[0], entry[1], entry[3]
	    restraint_index, i, exp_protectionfactor = entry[0], entry[0], entry[3]	#05/2017 GYH
            self.add_protectionfactor_restraint(i, exp_protectionfactor, model_protectionfactor=None)	# need to be fixed with add_protectionfactor_restraint function GYH 03/2017

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_pf()

    def save_expdata(self, filename):

        fout = file(filename, 'w')
        yaml.dump(data, fout)


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
#	print 'resid', i
	if model_protectionfactor == None:
		model_protectionfactor = np.zeros((len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0), len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)))	#GYH 03/2017
#		print len(self.allowed_xhs)
		for o in range(len(self.allowed_xcs)):
#			print o
                	for p in range(len(self.allowed_xhs)):
	                        for q in range(len(self.allowed_bs)):
#                                infile='Nc_x%0.1f_b%d_%03d.npy'(self.allowed_xcs[o], self.allowed_bs[q]) # will be modified based on input file fromat
#                                	Nc[self.allowed_xcs[o],self.allowed_bs[q]] = list(np.load(infile_Nc))      # will be modified based on input file fromat
#					Nh[self.allowed_xhs[p],self.allowed_bs[q]] = list(np.load(infile_Nh))
			#		Nc = Ncs[o,q]
			#		Nh = Nhs[p,q]
                                	for m in range(len(self.allowed_beta_c)):
                                        	for j in range(len(self.allowed_beta_h)):
                                                	for k in range(len(self.allowed_beta_0)):
			#					model_protectionfactor=compute_PF(self.allowed_beta_c[m], self.allowed_beta_h[j], self.allowed_beta_0[k], Nc, Nh)
#								model_protectionfactor[m,j,k,o,p,q] = 1
#								print 'allowed_beta_c[m]', self.allowed_beta_c[m], 'allowed_beta_h[j]', self.allowed_beta_h[j], 'allowed_beta_0[k]', self.allowed_beta_0[k], 'Ncs[o,q,i]', self.Ncs[o,q,i], 'Nhs[p,q,i]', self.Nhs[p,q,i]
                                                        	model_protectionfactor[m,j,k,o,p,q] = self.compute_PF(self.allowed_beta_c[m], self.allowed_beta_h[j], self.allowed_beta_0[k],self.Ncs[o,q,i], self.Nhs[p,q,i]) # GYH: will be modified with final file format 03/2017
#								print 'model_protectionfactor[',m,j,k,o,p,q,']', model_protectionfactor[m,j,k,o,p,q]
 
	self.protectionfactor_restraints.append(NMR_Protectionfactor(i, model_protectionfactor, exp_protectionfactor))	#???
	self.nprotectionfactor += 1

    def compute_PF(self, beta_c, beta_h, beta_0, Nc, Nh):
	"""Calculate predicted (ln PF)
	INPUT    (nres, 2) array with columns <N_c> and <N_h> for each residue, 
	OUTPUT   array of <ln PF> = beta_c <N_c> + beta_h <N_h> + beta_0 for all residues
	"""
	return beta_c * Nc + beta_h * Nh + beta_0  #GYH: new equation for PF calculation 03/2017 


    def build_groups_noe(self, verbose=False):
        """Build equivalency and ambiguity groups for distances and dihedrals,
        and store pre-computed SSE and d.o.f for distances and dihedrals"""

        # compile distance_equivalency_groups from the list of NMR_Distance() objects
        for i in range(len(self.distance_restraints)):
            d = self.distance_restraints[i]
            if d.equivalency_index != None:
                if not self.distance_equivalency_groups.has_key(d.equivalency_index):
                    self.distance_equivalency_groups[d.equivalency_index] = []
                self.distance_equivalency_groups[d.equivalency_index].append(i)
        if verbose:
            print 'self.distance_equivalency_groups', self.distance_equivalency_groups

        # NOTE: ambiguous group indices have already been compiled in load_exp_data()

        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()


        # precompute SSE and Ndof for distances
        self.compute_sse_distances()



    def build_groups_J(self, verbose=False):
        """Build equivalency and ambiguity groups for distances and dihedrals,
        and store pre-computed SSE and d.o.f for distances and dihedrals"""


        # compile dihedral_equivalency_groups from the list of NMR_Dihedral() objects
        for i in range(len(self.dihedral_restraints)):
            d = self.dihedral_restraints[i]
            # print 'd', d, 'd.equivalency_index', d.equivalency_index
            if d.equivalency_index != None:
                if not self.dihedral_equivalency_groups.has_key(d.equivalency_index):
                    self.dihedral_equivalency_groups[d.equivalency_index] = []
                self.dihedral_equivalency_groups[d.equivalency_index].append(i)
        if verbose:
            print 'self.dihedral_equivalency_groups', self.dihedral_equivalency_groups


        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()

        # precompute SSE and Ndof for dihedrals
        self.compute_sse_dihedrals()


    def build_groups_cs_H(self, verbose=False):
#        """Build equivalency and ambiguity groups for distances and dihedrals,
#        and store pre-computed SSE and d.o.f for distances and dihedrals"""


        # compile chemicalshift_equivalency_groups from the list of NMR_Chemicalshift() objects   #GYH
#        for i in range(len(self.chemicalshift_restraints)):
#            d = self.chemicalshift_restraints[i]
#            if d.equivalency_index != None:
#                if not self.chemicalshift_equivalency_groups.has_key(d.equivalency_index):
#                   self.chemicalshift_equivalency_groups[d.equivalency_index] = []
#                self.chemicalshift_equivalency_groups[d.equivalency_index].append(i)
#        if (1):
#            print 'self.chemicalshift_equivalency_groups', self.chemicalshift_equivalency_groups

        # precompute SSE and Ndof for chemical shift #GYH
        self.compute_sse_chemicalshift_H()

    def build_groups_cs_Ha(self, verbose=False):
#        """Build equivalency and ambiguity groups for distances and dihedrals,
#        and store pre-computed SSE and d.o.f for distances and dihedrals"""


        # compile chemicalshift_equivalency_groups from the list of NMR_Chemicalshift() objects   #GYH
#        for i in range(len(self.chemicalshift_restraints)):
#            d = self.chemicalshift_restraints[i]
#            if d.equivalency_index != None:
#                if not self.chemicalshift_equivalency_groups.has_key(d.equivalency_index):
#                   self.chemicalshift_equivalency_groups[d.equivalency_index] = []
#                self.chemicalshift_equivalency_groups[d.equivalency_index].append(i)
#        if (1):
#            print 'self.chemicalshift_equivalency_groups', self.chemicalshift_equivalency_groups

        # precompute SSE and Ndof for chemical shift #GYH
        self.compute_sse_chemicalshift_Ha()


    def build_groups_cs_N(self, verbose=False):
#        """Build equivalency and ambiguity groups for distances and dihedrals,
#        and store pre-computed SSE and d.o.f for distances and dihedrals"""


        # compile chemicalshift_equivalency_groups from the list of NMR_Chemicalshift() objects   #GYH
#        for i in range(len(self.chemicalshift_restraints)):
#            d = self.chemicalshift_restraints[i]
#            if d.equivalency_index != None:
#                if not self.chemicalshift_equivalency_groups.has_key(d.equivalency_index):
#                   self.chemicalshift_equivalency_groups[d.equivalency_index] = []
#                self.chemicalshift_equivalency_groups[d.equivalency_index].append(i)
#        if (1):
#            print 'self.chemicalshift_equivalency_groups', self.chemicalshift_equivalency_groups

        # precompute SSE and Ndof for chemical shift #GYH
        self.compute_sse_chemicalshift_N()

    def build_groups_cs_Ca(self, verbose=False):
#        """Build equivalency and ambiguity groups for distances and dihedrals,
#        and store pre-computed SSE and d.o.f for distances and dihedrals"""


        # compile chemicalshift_equivalency_groups from the list of NMR_Chemicalshift() objects   #GYH
#        for i in range(len(self.chemicalshift_restraints)):
#            d = self.chemicalshift_restraints[i]
#            if d.equivalency_index != None:
#                if not self.chemicalshift_equivalency_groups.has_key(d.equivalency_index):
#                   self.chemicalshift_equivalency_groups[d.equivalency_index] = []
#                self.chemicalshift_equivalency_groups[d.equivalency_index].append(i)
#        if (1):
#            print 'self.chemicalshift_equivalency_groups', self.chemicalshift_equivalency_groups

        # precompute SSE and Ndof for chemical shift #GYH
        self.compute_sse_chemicalshift_Ca()

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

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on their equivalency group."""

        for group in self.distance_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.distance_restraints[i].weight = 1.0/n 

        for group in self.dihedral_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.dihedral_restraints[i].weight = 1.0/n


    def compute_sse_distances(self, debug=False):
        """Returns the (weighted) sum of squared errors for distances,
        and the *effective* number of distances (i.e. the sum of the weights)"""

        for g in range(len(self.allowed_gamma)):

            sse = 0.
            N = 0.
            for i in range(self.ndistances):

                gamma = self.allowed_gamma[g]
                if g == 0:
                    print '---->', i, '(%d,%d)'%(self.distance_restraints[i].i, self.distance_restraints[i].j),
                    print '      exp',  self.distance_restraints[i].exp_distance, 'model', self.distance_restraints[i].model_distance
                if self.use_log_normal_distances:
                    err = np.log(self.distance_restraints[i].model_distance/(gamma*self.distance_restraints[i].exp_distance))
                    #print 'log-normal err', err
                else:
                    err = gamma*self.distance_restraints[i].exp_distance - self.distance_restraints[i].model_distance
                    #print 'err', err
                sse += (self.distance_restraints[i].weight * err**2.0)
                N += self.distance_restraints[i].weight
            #print 'total sse =', sse
            self.sse_distances[g] = sse 
            self.Ndof_distances = N
        if debug:
            print 'self.sse_distances', self.sse_distances

    def compute_sse_dihedrals(self, debug=False):
        """Returns the (weighted) sum of squared errors for J-coupling values,
        and the *effective* number of distances (i.e. the sum of the weights)"""

        sse = 0.0
        N =  0.0

        remaining_dihedral_indices = range(len(self.dihedral_restraints))

        # First, find all the equivalent groups, and average them.
        # (We assume that all of the experimental values have set to the same J value)
        for equivalency_index, group in self.dihedral_equivalency_groups.iteritems():
            avgJ = np.array([self.dihedral_restraints[i].model_Jcoupling for i in group]).mean()
            err = self.dihedral_restraints[group[0]].exp_Jcoupling - avgJ
            if debug:
                print group, 'avgJ_model',avgJ, 'expJ',self.dihedral_restraints[group[0]].exp_Jcoupling, 'err', err
            sse += err**2.0 
            N += 1
            # remove group indices from remaining_dihedral_indices 
            for i in group:
                remaining_dihedral_indices.remove(i) 

        for i in remaining_dihedral_indices:
            err = self.dihedral_restraints[i].exp_Jcoupling - self.dihedral_restraints[i].model_Jcoupling
            if debug:
                print 'J_model', self.dihedral_restraints[i].model_Jcoupling, 'exp', self.dihedral_restraints[i].exp_Jcoupling, 'err', err
            sse += (self.dihedral_restraints[i].weight * err**2.0)
            N += self.dihedral_restraints[i].weight

        if debug:
            print 'total sse', sse
        self.sse_dihedrals = sse 
        self.Ndof_dihedrals = N

    def compute_sse_chemicalshift_H(self, debug=False):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_H = 0.0
        N_H = 0.0
	for i in range(self.nchemicalshift_H):

#		print '---->', i, '%d'%self.chemicalshift_restraints[i].i,
#        	print '      exp', self.chemicalshift_restraints[i].exp_chemicalshift, 'model', self.chemicalshift_restraints[i].model_chemicalshift

                err_H=self.chemicalshift_H_restraints[i].model_chemicalshift_H - self.chemicalshift_H_restraints[i].exp_chemicalshift_H
                sse_H += (self.chemicalshift_H_restraints[i].weight*err_H**2.0)
                N_H += self.chemicalshift_H_restraints[i].weight
        self.sse_chemicalshift_H = sse_H
        self.Ndof_chemicalshift_H = N_H
        if debug:
            print 'self.sse_chemicalshift_H', self.sse_chemicalshift_H

    def compute_sse_chemicalshift_Ha(self, debug=False):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_Ha = 0.0
        N_Ha = 0.0
        for i in range(self.nchemicalshift_Ha):

#               print '---->', i, '%d'%self.chemicalshift_restraints[i].i,
#               print '      exp', self.chemicalshift_restraints[i].exp_chemicalshift, 'model', self.chemicalshift_restraints[i].model_chemicalshift

                err_Ha=self.chemicalshift_Ha_restraints[i].model_chemicalshift_Ha - self.chemicalshift_Ha_restraints[i].exp_chemicalshift_Ha
                sse_Ha += (self.chemicalshift_Ha_restraints[i].weight*err_Ha**2.0)
#                print self.chemicalshift_Ha_restraints[i].weight
                N_Ha += self.chemicalshift_Ha_restraints[i].weight
        self.sse_chemicalshift_Ha = sse_Ha
        self.Ndof_chemicalshift_Ha = N_Ha
        if debug:
            print 'self.sse_chemicalshift_Ha', self.sse_chemicalshift_Ha


    def compute_sse_chemicalshift_N(self, debug=False):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_N = 0.0
        N_N = 0.0
        for i in range(self.nchemicalshift_N):

#               print '---->', i, '%d'%self.chemicalshift_restraints[i].i,
#               print '      exp', self.chemicalshift_restraints[i].exp_chemicalshift, 'model', self.chemicalshift_restraints[i].model_chemicalshift

                err_N=self.chemicalshift_N_restraints[i].model_chemicalshift_N - self.chemicalshift_N_restraints[i].exp_chemicalshift_N
                sse_N += (self.chemicalshift_N_restraints[i].weight*err_N**2.0)
#                print self.chemicalshift_N_restraints[i].weight
                N_N += self.chemicalshift_N_restraints[i].weight
        self.sse_chemicalshift_N = sse_N
        self.Ndof_chemicalshift_N = N_N
        if debug:
            print 'self.sse_chemicalshift_N', self.sse_chemicalshift_N


    def compute_sse_chemicalshift_Ca(self, debug=False):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse_Ca = 0.0
        N_Ca = 0.0
        for i in range(self.nchemicalshift_Ca):

#               print '---->', i, '%d'%self.chemicalshift_restraints[i].i,
#               print '      exp', self.chemicalshift_restraints[i].exp_chemicalshift, 'model', self.chemicalshift_restraints[i].model_chemicalshift

                err_Ca=self.chemicalshift_Ca_restraints[i].model_chemicalshift_Ca - self.chemicalshift_Ca_restraints[i].exp_chemicalshift_Ca
                sse_Ca += (self.chemicalshift_Ca_restraints[i].weight*err_Ca**2.0)
#                print self.chemicalshift_Ca_restraints[i].weight
                N_Ca += self.chemicalshift_Ca_restraints[i].weight
        self.sse_chemicalshift_Ca = sse_Ca
        self.Ndof_chemicalshift_Ca = N_Ca
        if debug:
            print 'self.sse_chemicalshift_Ca', self.sse_chemicalshift_Ca


    def compute_sse_protectionfactor(self,debug=False):         #GYH
#    """ new defined sse based on computed PF"""
        self.sse_protectionfactor = np.zeros((len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0), len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)))
	for o in range(len(self.allowed_xcs)):
        	for p in range(len(self.allowed_xhs)):
                	for q in range(len(self.allowed_bs)):
                        	for m in range(len(self.allowed_beta_c)):
                                	for j in range(len(self.allowed_beta_h)):
                                        	for k in range(len(self.allowed_beta_0)):
							sse = 0.
							N = 0.
							for i in range(self.nprotectionfactor):
								err=self.protectionfactor_restraints[i].model_protectionfactor[m,j,k,o,p,q] - self.protectionfactor_restraints[i].exp_protectionfactor
								sse += (self.protectionfactor_restraints[i].weight * err**2.0)
								N += self.protectionfactor_restraints[i].weight
							self.sse_protectionfactor[m,j,k,o,p,q] = sse
							self.Ndof_protectionfactor = N #should equal to number of residues
#    sys.exit()

#    def compute_sse_protectionfactor(self,debug=False):		#GYH
#	"""Returns the (weighted) sum of squared errors for protection factor values"""
#	
#	sse = 0.0
#	N = 0.0
#	for i in range(self.nprotectionfactor):
#				print '---->', i, '(%d)'%(self.protectionfactor_restraints[i].i),
#	                        print '      exp',  self.protectionfactor_restraints[i].exp_protectionfactor, 'model', self.protectionfactor_restraints[i].model_protectionfactor
#			
#		err=self.protectionfactor_restraints[i].model_protectionfactor - self.protectionfactor_restraints[i].exp_protectionfactor
#                        err=self.protectionfactor_restraints[i].model_protectionfactor - self.protectionfactor_restraints[i].exp_protectionfactor
#		sse += (self.protectionfactor_restraints[i].weight * err**2.0)
#		N += self.protectionfactor_restraints[i].weight 
#	self.sse_protectionfactor = sse
#	self.Ndof_protectionfactor = N
#	if debug:
#	    print 'self.sse_protectionfactor', self.sse_protectionfactor


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
#	print '(self.nprotectionfactor)', (self.nprotectionfactor)
        self.neglog_reference_priors_PF= np.zeros((self.nprotectionfactor, len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0), len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)))
        self.sum_neglog_reference_priors_PF = 0.
        for j in range(self.nprotectionfactor):	# number of residues
		for o in range(len(self.allowed_xcs)):
                	for p in range(len(self.allowed_xhs)):
	                        for q in range(len(self.allowed_bs)):
                                	for m in range(len(self.allowed_beta_c)):
                                        	for n in range(len(self.allowed_beta_h)):
                                                	for k in range(len(self.allowed_beta_0)):
#								self.neglog_reference_priors_PF[j,m,n,k,o,p,q] = np.max(-1.0*self.protectionfactor_restraints[j].model_protectionfactor[m,n,k,o,p,q],0.0)
#		self.neglog_reference_priors_PF[j,m,n,k,o,p,q] = np.max(-1.0*self.protectionfactor_restraints[j].model_protectionfactor,0.0)
						                self.neglog_reference_priors_PF[j,m,n,k,o,p,q] = np.log(self.betas_PF[j]) + (self.protectionfactor_restraints[j].model_protectionfactor[m,n,k,o,p,q])/self.betas_PF[j]
#								print j, 'np.log(self.betas_PF[j])', np.log(self.betas_PF[j]), '(self.protectionfactor_restraints[j].model_protectionfactor[m,n,k,o,p,q])', (self.protectionfactor_restraints[j].model_protectionfactor[m,n,k,o,p,q]), 'self.neglog_reference_priors_PF[j,m,n,k,o,p,q]', self.neglog_reference_priors_PF[j,m,n,k,o,p,q]
						                self.sum_neglog_reference_priors_PF  += self.protectionfactor_restraints[j].weight * self.neglog_reference_priors_PF[j,m,n,k,o,p,q]


    def compute_gaussian_neglog_reference_priors_PF(self):     #GYH
        """An alternative option for reference potential based on Gaussian distribution"""
        self.gaussian_neglog_reference_priors_PF = np.zeros((self.nprotectionfactor, len(self.allowed_beta_c), len(self.allowed_beta_h), len(self.allowed_beta_0), len(self.allowed_xcs), len(self.allowed_xhs), len(self.allowed_bs)))
        self.sum_gaussian_neglog_reference_priors_PF = 0.
#        for j in range(self.nprotectionfactor):
#	    print j, 'self.ref_sigma_PF[j]', self.ref_sigma_PF[j], 'self.ref_mean_PF[j]', self.ref_mean_PF[j] 
#            self.gaussian_neglog_reference_priors_PF[j] = np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_PF[j]) + (self.protectionfactor_restraints[j].model_protectionfactor - self.ref_mean_PF[j])**2.0/(2*self.ref_sigma_PF[j]**2.0)
#            self.sum_gaussian_neglog_reference_priors_PF += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_priors_PF[j]

        for j in range(self.nprotectionfactor):
                for o in range(len(self.allowed_xcs)):
                        for p in range(len(self.allowed_xhs)):
                                for q in range(len(self.allowed_bs)):
                                        for m in range(len(self.allowed_beta_c)):
                                                for n in range(len(self.allowed_beta_h)):
                                                        for k in range(len(self.allowed_beta_0)):
								self.gaussian_neglog_reference_priors_PF[j,o,p,q,m,n,k]=np.log(np.sqrt(2.0*np.pi)) + np.log(self.ref_sigma_PF[j]) + (self.protectionfactor_restraints[j].model_protectionfactor[m,n,k,o,p,q] - self.ref_mean_PF[j])**2.0/(2*self.ref_sigma_PF[j]**2.0)
								self.sum_gaussian_neglog_reference_priors_PF += self.protectionfactor_restraints[j].weight * self.gaussian_neglog_reference_priors_PF[m,n,k,o,p,q]

    def switch_distances(self, indices1, indices2):
        """Given two lists of ambiguous distance pair indices, switch their distances and recompute the sum of squared errors (SSE)."""
        distance1 = self.distance_restraints[indices1[0]].exp_distance 
        distance2 = self.distance_restraints[indices2[0]].exp_distance
        for i in indices1:
            self.distance_restraints[i].exp_distance = distance2
        for j in indices2:
            self.distance_restraints[j].exp_distance = distance1
        self.compute_sse_distances()



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




class NMR_Distance(object):
    """A class to store NMR distance information."""

    def __init__(self, i, j, model_distance, exp_distance, equivalency_index=None, ambiguity_index=None):

        # Atom indices from the Conformation() defining this distance
        self.i = i
        self.j = j  

        # the model distance in this structure (in Angstroms)
        self.model_distance = model_distance 

        # the experimental NOE distance (in Angstroms)
        self.exp_distance = exp_distance 

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1

        # the index of the ambiguity group (i.e. some groups distances have
        # distinct values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index

        
class NMR_Dihedral(object):
    """A class to store NMR J-coupling dihedral information."""

    def __init__(self, i, j, k, l, model_Jcoupling, exp_Jcoupling, model_angle, equivalency_index=None, ambiguity_index=None):

        # Atom indices from the Conformation() defining this dihedral
        self.i = i
        self.j = j
        self.k = k
        self.l = l

        # the model distance in this structure (in Angstroms)
        self.model_Jcoupling = model_Jcoupling

        # the experimental J-coupling constant 
        self.exp_Jcoupling = exp_Jcoupling

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1

        # the index of the ambiguity group (i.e. some groups distances have
        # distant values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index

class NMR_Chemicalshift_H(object):        #GYH
    """A class to store NMR chemical shift information."""

    def __init__(self, i, model_chemicalshift_H, exp_chemicalshift_H):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_H = model_chemicalshift_H

        # the experimental chemical shift 
        self.exp_chemicalshift_H = exp_chemicalshift_H

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0/3.0 # default is N=1




class NMR_Chemicalshift_Ha(object):        #GYH
    """A class to store NMR chemical shift information."""

    def __init__(self, i, model_chemicalshift_Ha, exp_chemicalshift_Ha):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_Ha = model_chemicalshift_Ha

        # the experimental chemical shift 
        self.exp_chemicalshift_Ha = exp_chemicalshift_Ha

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0/3.0 # default is N=1


class NMR_Chemicalshift_N(object):        #GYH
    """A class to store NMR chemical shift information."""

    def __init__(self, i, model_chemicalshift_N, exp_chemicalshift_N):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_N = model_chemicalshift_N

        # the experimental chemical shift 
        self.exp_chemicalshift_N = exp_chemicalshift_N

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0/3.0 # default is N=1


class NMR_Chemicalshift_Ca(object):        #GYH
    """A class to store NMR chemical shift information."""

    def __init__(self, i, model_chemicalshift_Ca, exp_chemicalshift_Ca):

        # Atom indices from the Conformation() defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model_chemicalshift_Ca = model_chemicalshift_Ca

        # the experimental chemical shift 
        self.exp_chemicalshift_Ca = exp_chemicalshift_Ca

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0/3.0 # default is N=1




class NMR_Protectionfactor(object):        #GYH
    """A class to store NMR protection factor information."""

    def __init__(self, i, model_protectionfactor, exp_protectionfactor):
        # Atom indices from the Conformation() defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model_protectionfactor = model_protectionfactor

        # the experimental protection factor 
        self.exp_protectionfactor = exp_protectionfactor


        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1


