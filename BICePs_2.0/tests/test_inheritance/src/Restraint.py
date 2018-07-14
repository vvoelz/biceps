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
import mdtraj
import yaml                       # Yet Another Multicolumn Layout

from KarplusRelation import *     # Class - returns J-coupling values from dihedral angles
from prep_cs import *    # Class - creates Chemical shift restraint file
from prep_noe import *   # Class - creates NOE (Nuclear Overhauser effect) restraint file
from prep_J import *     # Class - creates J-coupling const. restraint file
from prep_pf import *	  # Class - creates Protection factor restraint file   #GYH
from inheri import *
from toolbox import *
##############################################################################
# Code
##############################################################################

class Restraint(inheri):
    """A class to store a molecular structure, its complete set of
    experimental NOE, J-coupling, chemical shift and protection factor data, and
    Each Instances of this object"""

    def __init__(self, PDB_filename, lam, free_energy, data = None,
            use_log_normal_noe=False, dloggamma=np.log(1.01), gamma_min=0.2,
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
        self.use_log_normal_noe = use_log_normal_noe

	# initialize all restraint child class
	restraint_noe(gamma_min=self.gamma_min,gamma_max=self.gamma_max,dloggamma=self.dloggamma,use_log_normal_noe=self.use_log_normal_noe)	


        # Create a KarplusRelation object
        self.karplus = KarplusRelation()


        # If an experimental data file is given, load in the information
	if data != None:
		for i in data:
#			print i
			if i.endswith('.noe'):
				self.load_data_noe(i)
                        elif i.endswith('.J'):
				self.load_data_J(i)
                        elif i.endswith('.cs_H'):
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



    # Save Experimental Data: ### yaml ### {{{
    def save_expdata(self, filename):

        fout = file(filename, 'w')
        yaml.dump(data, fout)



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


