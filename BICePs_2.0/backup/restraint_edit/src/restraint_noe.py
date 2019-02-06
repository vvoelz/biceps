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
class restraint_noe(object):
    #Notes:# {{{
    '''

    '''
    # }}}
    def __init__(self,dlogsigma_noe=np.log(1.01),sigma_noe_min=0.05,
            sigma_noe_max=120.0,dloggamma=np.log(1.01),
            gamma_min=0.2,gamma_max=10.0):

        # Store distance restraint info
        self.distance_restraints = []
        self.distance_equivalency_groups = {}
        self.ambiguous_groups = []  # list of pairs of group indices, e.g.:   [ [[1,2,3],[4,5,6]],   [[7],[8]], ...]
        self.ndistances = 0
        # pick initial values for sigma_noe (std of experimental uncertainty in NOE distances)
        self.dlogsigma_noe = dlogsigma_noe  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.sigma_noe_min = sigma_noe_min
        self.sigma_noe_max = sigma_noe_max

        self.allowed_sigma_noe = np.exp(np.arange(np.log(self.sigma_noe_min), np.log(self.sigma_noe_max), self.dlogsigma_noe))
#        print 'self.allowed_sigma_noe', self.allowed_sigma_noe
#        print 'len(self.allowed_sigma_noe) =', len(self.allowed_sigma_noe)
        self.sigma_noe_index = len(self.allowed_sigma_noe)/2    # pick an intermediate value to start with
        self.sigma_noe = self.allowed_sigma_noe[self.sigma_noe_index]
        # pick initial values for gamma^(-1/6) (NOE scaling parameter)
        self.dloggamma = dloggamma  # stepsize in log(sigma_noe) - i.e. grow/shrink multiplier
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.allowed_gamma = np.exp(np.arange(np.log(self.gamma_min), np.log(self.gamma_max), self.dloggamma))
#        print 'self.allowed_gamma', self.allowed_gamma
#        print 'len(self.allowed_gamma) =', len(self.allowed_gamma)
        self.gamma_index = len(self.allowed_gamma)/2    # pick an intermediate value to start with
        self.gamma = self.allowed_gamma[self.gamma_index]




    """A class for all restraints"""

    # Load Experimental Data (ALL Restraints):{{{
    def load_data_noe(self, filename, verbose=False):
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
            restraint_index, i, j, exp_distance, model_distance = entry[0], entry[1], entry[4], entry[7], entry[8]
            self.add_distance_restraint(i, j, exp_distance, model_distance, equivalency_index=restraint_index)

        # build groups of equivalency group indices, ambiguous group etc.
        self.build_groups_noe()




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


    def adjust_weights(self):
        """Adjust the weights of distance and dihedral restraints based on their equivalency group."""

        for group in self.distance_equivalency_groups.values():
            n = float(len(group))
            for i in group:
                self.distance_restraints[i].weight = 1.0/n


    def compute_sse_distances(self, debug=False):
        """Returns the (weighted) sum of squared errors for distances,
        and the *effective* number of distances (i.e. the sum of the weights)"""

        for g in range(len(self.allowed_gamma)):

            sse = 0.
            N = 0.
            for i in range(self.ndistances):

                gamma = self.allowed_gamma[g]
#                if g == 0:
#                    print '---->', i, '(%d,%d)'%(self.distance_restraints[i].i, self.distance_restraints[i].j),
#                    print '      exp',  self.distance_restraints[i].exp_distance, 'model', self.distance_restraints[i].model_distance
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

class NMR_Distance(object):
    """A class to store NMR distance information."""

    # __init__:{{{
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



