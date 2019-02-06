import os, sys, glob
import numpy as np

#from msmbuilder import Conformation
import mdtraj
import yaml

from KarplusRelation import *
from RestraintFile_cs import *
from RestraintFile_noe import *
from RestraintFile_J import *
from RestraintFile_PF import *		#GYH


class Structure(object):
    """A class to store a molecular structure, its complete set of
    experimental NOE, J-coupling and chemical shift data, and   
    Each Instances of this obect"""

    def __init__(self, PDB_filename, free_energy, expdata_filename_noe=None, expdata_filename_J=None, expdata_filename_cs=None, expdata_filename_PF=None, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=10.0):
        """Initialize the class.
        INPUTS
	conf		A molecular structure as an msmbuilder Conformation() object.
                        NOTE: For cases where the structure is an ensemble (say, from clustering)
                        and the modeled NOE distances and coupling constants are averaged, 
                        the structure itself can just be a placeholder with the right atom names
                        and numbering
              
        free_energy     The (reduced) free energy f = beta*F of this conformation
        """

        self.PDB_filename = PDB_filename
    	self.expdata_filename_noe = expdata_filename_noe
    #    self.expdata_filename_J = expdata_filename_J
        self.expdata_filename_cs = expdata_filename_cs
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
        self.chemicalshift_restraints = []
        self.chemicalshift_equivalency_groups = {}
        self.chemicalshift_ambiguity_groups = {}
        self.nchemicalshift = 0

	# Store protection factor restraint info	#GYH
	self.protectionfactor_restraints = []
	self.protectionfactor_equivalency_groups = {}
	self.protectionfactor_ambiguity_groups = {}
	self.nprotectionfactor = 0

        # Create a KarplusRelation object
        self.karplus = KarplusRelation()

        # variables to store pre-computed SSE and effective degrees of freedom (d.o.f.)
        self.sse_distances = np.array([0.0 for gamma in self.allowed_gamma])
        self.Ndof_distances = None
        self.sse_dihedrals = None
        self.Ndof_dihedrals = None
        self.sse_chemicalshift = None #GYH
        self.Ndof_chemicalshift = None  #GYH
	self.sse_protectionfactor = None #GYH
	self.Ndof_protectionfactor = None #GYH
        self.betas = None   # if reference is used, an array of N_j betas for each distance
        self.neglog_reference_priors = None
        self.sum_neglog_reference_priors = 0.0

        # If an experimental data file is given, load in the information
	self.expdata_filename_noe = expdata_filename_noe
        if expdata_filename_noe != None:
                self.load_expdata_noe(expdata_filename_noe)
        self.expdata_filename_J = expdata_filename_J
#        if expdata_filename_J != None:
#                self.load_expdata_J(expdata_filename_J)
	self.expdata_filename_cs = expdata_filename_cs
	if expdata_filename_cs != None:
		self.load_expdata_cs(expdata_filename_cs)	#GYH        
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
        self.build_groups()


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

        # add the Jcoupling restraints
        for entry in data:
            restraint_index, i, j, k, l, exp_Jcoupling, karplus  = entry[0], entry[1], entry[4], entry[7], entry[10], entry[13], entry[14]
            self.add_dihedral_restraint(i, j, k, l, exp_Jcoupling, model_Jcoupling=None, equivalency_index=None, karplus_key=karplus)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()



    def load_expdata_cs(self, filename, verbose=False):
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

        # the equivalency indices for distances are in the first column of the *.biceps file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'chemicalshift equivalency_indices', equivalency_indices

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

        # add the chemical shift restraints
        for entry in data:
            restraint_index, i, exp_chemicalshift = entry[0], entry[1], entry[4]
            self.add_chemicalshift_restraint(i, exp_chemicalshift, model_chemicalshift=None, equivalency_index=restraint_index)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()


    def load_expdata_PF(self, filename, verbose=False):
        """Load in the experimental protection factor restraints from a .PF file format.
        """

        # Read in the lines of the chemicalshift data file
        b = RestraintFile_PF(filename=filename)
        if verbose:
                print b.lines
        data = []
        for line in b.lines:
                data.append( b.parse_line_PF(line) )  # [restraint_index, atom_index1, res1, atom_name1, protectionfactor] #GYH: need adjust once data are available!!!

        if verbose:
            print 'Loaded from', filename, ':'
            for entry in data:
                print entry

        ### distances ###

        # the equivalency indices for protection factors are in the first column of the *.PF file
        equivalency_indices = [entry[0] for entry in data]
        if verbose:
            print 'protectionfactor equivalency_indices', equivalency_indices

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

        # add the protection factor restraints
        for entry in data:				#GYH
            restraint_index, i, exp_protectionfactor = entry[0], entry[1], entry[4]
            self.add_protectionfactor_restraint(i, exp_protectionfactor, model_protectionfactor=None, equivalency_index=restraint_index)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()




    def load_expdata_yaml(self, filename, verbose=False):
        """Load in the experimental NOE distances from and J-coupling data from a YAML file format."""

        # Read in the YAML data as a dictionary
        stream = file(filename, 'r')
        data = yaml.load(stream)

        if verbose:
            print 'Loaded from', filename, ':'
            for key, value in data.iteritems():
                print key, value

        ### distances ###

        # compile equivalency indices for distances
        equivalency_indices = [None for pair in data['NOE_PairIndex']]
        for e in range(len(data['NOE_Equivalent'])):
            thisgroup = data['NOE_Equivalent'][e]
            for pair in thisgroup: 
                # find the index of the pair in the NOE_PairIndex pairlist, ... and tag it with the index of the equivalency group 
                pair_index = data['NOE_PairIndex'].index(pair) 
                # ... and tag it with the index of the equivalency group 
                equivalency_indices[pair_index] = e
        if verbose:
            print 'distance equivalency_indices', equivalency_indices

        # compile ambiguity indices for distances
        try:
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

        # add the distance restraints
        for pair_index in range(len(data['NOE_PairIndex'])):
            i, j = data['NOE_PairIndex'][pair_index]
            exp_distance = data['NOE_Expt'][pair_index]
            e = equivalency_indices[pair_index]
            self.add_distance_restraint(i, j, exp_distance, model_distance=None, equivalency_index=e) 

        ### dihedrals ###

        # compile equivalency indices for dihedrals
        equivalency_indices = [None for quadruple in data['DihedralIndex']]
        for e in range(len(data['Dihedral_Equivalent'])):
            thisgroup = data['Dihedral_Equivalent'][e]
            print 'thisgroup', thisgroup
            for quadruple in thisgroup:
                print '\tquadruple', quadruple, 'e', e
                # find the index of the quadruple in the DihedralIndex quaduple-list, ... and tag it with the index of the equivalency group 
                quadruple_index = data['DihedralIndex'].index(quadruple)
                # ... and tag it with the index of the equivalency group 
                equivalency_indices[quadruple_index] = e
        if verbose:
            print 'dihedral equivalency_indices', equivalency_indices

        # add the dihedral restraints
        for quadruple_index in range(len(data['DihedralIndex'])):
            i, j, k, l = data['DihedralIndex'][quadruple_index]
            exp_Jcoupling = data['JCoupling_Expt'][quadruple_index]
            karplus_key = data['JCoupling_Karplus'][quadruple_index]
            e = equivalency_indices[quadruple_index]
            self.add_dihedral_restraint(i, j, k, l, exp_Jcoupling, model_Jcoupling=None, equivalency_index=e, karplus_key=karplus_key)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups()


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



    def add_chemicalshift_restraint(self, i, exp_chemicalshift, model_chemicalshift=None, equivalency_index=None):
        """Add a chemicalshift NMR_Chemicalshift() object to the list."""
         # if the modeled distance is not specified, compute the distance from the conformation
#       Ind = self.conf.topology.select("index == j")
#       t=self.conf.atom_slice(Ind)
        if model_chemicalshift == None:
 #              r=md.nmr.chemicalshifts_shiftx2(r,pH=2.5, temperature = 280.0)    

 #              model_chemicalshift = r.mean(axis=1)
                model_chemicalshift = 1  # will be replaced by pre-computed cs
        self.chemicalshift_restraints.append( NMR_Chemicalshift(i, model_chemicalshift, exp_chemicalshift, equivalency_index=equivalency_index))

        self.nchemicalshift += 1

    def add_protectionfactor_restraint(self, i, exp_protectionfactor, model_protectionfactor=None, equivalency_index=None):
	"""Add a protectionfactor NMR_Protectionfactor() object to the list."""
	if model_protectionfactor == None:
		model_protectionfactor = 1 #GYH: will be replaced by pre-computed PF
	self.protectionfactor_restraints.append(NMR_Protectionfactor(i, model_protectionfactor, exp_protectionfactor, equivalency_index=equivalency_index))
	self.nprotectionfactor += 1

    def build_groups(self, verbose=False):
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

        # compile chemicalshift_equivalency_groups from the list of NMR_Chemicalshift() objects   #GYH
        for i in range(len(self.chemicalshift_restraints)):
            d = self.chemicalshift_restraints[i]
            if d.equivalency_index != None:
                if not self.chemicalshift_equivalency_groups.has_key(d.equivalency_index):
                   self.chemicalshift_equivalency_groups[d.equivalency_index] = []
                self.chemicalshift_equivalency_groups[d.equivalency_index].append(i)
        if verbose:
            print 'self.chemicalshift_equivalency_groups', self.chemicalshift_equivalency_groups

	# compile protectionfactor_equivalency_groups from the list of NMR_Protectionfactor() objects	#GYH
	for i in range(len(self.protectionfactor_restraints)):
	    d = self.protectionfactor_restraints[i]
	    if d.equivalency_index != None:
		if not self.protectionfactor_equivalency_groups.has_key(d.equivalency_index):
		   self.protectionfactor_equivalency_groups[d.equivalency_index] = []
		self.protectionfactor_equivalency_groups[d.equivalency_index].append(i)
	if verbose:
	    print 'self.protectionfactor_equivalency_groups', self.protectionfactor_equivalency_groups


        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()

        # precompute SSE and Ndof for distances
        self.compute_sse_distances()

        # precompute SSE and Ndof for dihedrals
        self.compute_sse_dihedrals()

        # precompute SSE and Ndof for chemical shift #GYH
        self.compute_sse_chemicalshift()
     
	# precompute SSE and Ndof for protection factor #GYH
	self.compute_sse_protectionfactor()

    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on their equivalency group."""

        for group in self.distance_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.distance_restraints[i].weight = 2.0/n 

        for group in self.dihedral_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.dihedral_restraints[i].weight = 2.0/n

        for group in self.chemicalshift_equivalency_groups.values():    #GYH
            n = float(len(group))
            for i in group:
                self.chemicalshift_restraints[i].weight = 1.0/n

	for group in self.protectionfactor_equivalency_groups.values():		#GYH
	    n = float(len(group))
	    for i in group:
		self.protectionfactor_restraints[i].weight = 1.0/n

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

    def compute_sse_chemicalshift(self, debug=False):    #GYH
        """Returns the (weighted) sum of squared errors for chemical shift values"""
#       for g in range(len(self.allowed_gamma)):

        sse = 0.0
        N = 0.0
	for i in range(self.nchemicalshift):

#		print '---->', i, '%d'%self.chemicalshift_restraints[i].i,
#        	print '      exp', self.chemicalshift_restraints[i].exp_chemicalshift, 'model', self.chemicalshift_restraints[i].model_chemicalshift

                err=self.chemicalshift_restraints[i].model_chemicalshift - self.chemicalshift_restraints[i].exp_chemicalshift
                sse += (self.chemicalshift_restraints[i].weight * err**2.0)
                N += self.chemicalshift_restraints[i].weight
        self.sse_chemicalshift = sse
        self.Ndof_chemicalshift = N
        if debug:
            print 'self.sse_chemicalshift', self.sse_chemicalshift

    def compute_sse_protectionfactor(self,debug=False):		#GYH
	"""Returns the (weighted) sum of squared errors for protection factor values"""
	sse = 0.0
	N = 0.0
	for i in range(self.nprotectionfactor):
		err=self.protectionfactor_restraints[i].model_protectionfactor - self.protectionfactor_restraints[i].exp_protectionfactor
		sse += (self.protectionfactor_restraints[i].weight * err**2.0)
		N += self.protectionfactor_restraints[i].weight
	self.sse_protectionfactor = sse
	self.Ndof_protectionfactor = N
	if debug:
	    print 'self.sse_protectionfactor', self.sse_protectionfactor


    def compute_neglog_reference_priors(self):
        """Uses the stored beta information (calculated across all structures) to calculate
        - log P_ref(distance[j) for each distance j."""

        # print 'self.betas', self.betas

        self.neglog_reference_priors = np.zeros(self.ndistances)
        self.sum_neglog_reference_priors = 0.
        for j in range(self.ndistances):
            self.neglog_reference_priors[j] = np.log(self.betas[j]) + self.distance_restraints[j].model_distance/self.betas[j]
            self.sum_neglog_reference_priors  += self.distance_restraints[j].weight * self.neglog_reference_priors[j]


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

class NMR_Chemicalshift(object):        #GYH
    """A class to store NMR chemical shift information."""

    def __init__(self, i, model_chemicalshift, exp_chemicalshift, equivalency_index=None, ambiguity_index=None):
        # Atom indices from the Conformation() defining this chemical shift
	self.i = i

        # the model chemical shift in this structure (in ppm)
	self.model_chemicalshift = model_chemicalshift

        # the experimental chemical shift 
	self.exp_chemicalshift = exp_chemicalshift

        # the index of the equivalency group (not likely in this case but just in case we need it in the future)
	self.equivalency_index = equivalency_index

        # N equivalent chemical shift should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
	self.weight = 1.0 # default is N=1

        # the index of the ambiguity group (not likely in this case but just in case we need it in the future)
	self.ambiguity_index = ambiguity_index

class NMR_Protectionfactor(object):        #GYH
    """A class to store NMR protection factor information."""

    def __init__(self, i, model_protectionfactor, exp_protectionfactor, equivalency_index=None, ambiguity_index=None):
        # Atom indices from the Conformation() defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model_protectionfactor = model_protectionfactor

        # the experimental protection factor 
        self.exp_protectionfactor = exp_protectionfactor

        # the index of the equivalency group (not likely in this case but just in case we need it in the future)
        self.equivalency_index = equivalency_index

        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1

        # the index of the ambiguity group (not likely in this case but just in case we need it in the future)
        self.ambiguity_index = ambiguity_index

