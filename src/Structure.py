import os, sys, glob
import numpy as np
import mdtraj
try:
    from msmbuilder import Conformation
except:
    pass
import yaml

from KarplusRelation import *


class Structure(object):
    """A class to store a molecular structure, its complete set of
    experimental NOE and J-coupling data, and   
    Each Instances of this obect"""

    def __init__(self, PDB_filename, free_energy, expdata_filename=None, use_log_normal_distances=False,
                       dloggamma=np.log(1.01), gamma_min=0.5, gamma_max=2.0):
        """Initialize the class.
        INPUTS
	    conf		    A molecular structure as an msmbuilder Conformation() object.
                        NOTE: For cases where the structure is an ensemble (say, from clustering)
                        and the modeled NOE distances and coupling constants are averaged, 
                        the structure itself can just be a placeholder with the right atom names
                        and numbering
              
        free_energy     The (reduced) free energy f = beta*F of this conformation
        """

        self.PDB_filename = PDB_filename
        self.expdata_filename = expdata_filename
        try:
            self.conf = Conformation.load_from_pdb(PDB_filename)
            # Convert the coordinates from nm to Angstrom units
            self.conf["XYZ"] = self.conf["XYZ"]*10.0
        except:
            try:
                self.conf = mdtraj.load_pdb(PDB_filename).xyz*10.0
            except:
                print "Cannot load pdb file. Either of msmbuilder.Conformation or mdtraj is needed."
                sys.exit()
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

        # Create a KarplusRelation object
        self.karplus = KarplusRelation()

        # variables to store pre-computed SSE and effective degrees of freedom (d.o.f.)
        self.sse_distances = np.array([0.0 for gamma in self.allowed_gamma])
        self.Ndof_distances = None
        self.sse_dihedrals = None
        self.Ndof_dihedrals = None
        self.betas = None   # if reference is used, an array of N_j betas for each distance
        self.neglog_reference_priors = None
        self.sum_neglog_reference_priors = 0.0

        # If an experimental data file is given, load in the information
        self.expdata_filename = expdata_filename
        if expdata_filename != None:
            self.load_expdata(expdata_filename)


    def load_expdata(self, filename, verbose=False):
        """Load in the experimental NOE distance and J-coupling data from a YAML file format."""

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
            try:
                ri = self.conf["XYZ"][i,:]
                rj = self.conf["XYZ"][j,:]
            except:
                ri = self.conf[0,i,:]
                rj = self.conf[0,j,:]
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
            try:
                ri, rj, rk, rl = [self.conf["XYZ"][x,:] for x in [i, j, k, l]]
            except:
                ri, rj, rk, rl = [self.conf[0,x,:] for x in [i,j,k,l]]
            model_angle = self.dihedral_angle(ri,rj,rk,rl)
         
            ###########################
            # NOTE: In the future, this function can be more sophisticated, parsing atom types
            # or karplus relation types on a case-by-case basis
            model_Jcoupling = self.karplus.J(model_angle, karplus_key) 

        self.dihedral_restraints.append( NMR_Dihedral(i, j, k, l, model_Jcoupling, exp_Jcoupling, model_angle,
                                                      equivalency_index=equivalency_index) )
        self.ndihedrals += 1



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

        # adjust the weights of distances and dihedrals to account for equivalencies
        self.adjust_weights()

        # precompute SSE and Ndof for distances
        self.compute_sse_distances()

        # precompute SSE and Ndof for dihedrals
        self.compute_sse_dihedrals()

     


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
                #print i, '(%d,%d)'%(self.distance_restraints[i].i, self.distance_restraints[i].j),
                #print 'exp',  self.distance_restraints[i].exp_distance, 'model', self.distance_restraints[i].model_distance,
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


