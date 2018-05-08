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
class restraint(object):
    #Notes:# {{{
    '''

    '''
    # }}}
    def __init__(self):
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
        self.nchemicalshift_H = 0
        self.nchemicalshift_Ha = 0
        self.nchemicalshift_N = 0
        self.nchemicalshift_Ca = 0

        # Store protection factor restraint info        #GYH
        self.protectionfactor_restraints = []
        self.nprotectionfactor = 0



    """A class for all restraints"""

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





