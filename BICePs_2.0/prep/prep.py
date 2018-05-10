import os, sys, glob, string
import numpy as np
import mdtraj as md
import re

sys.path.append('src/')
from prep_cs import *
from prep_J import *
from prep_noe import *
from prep_pf import *

class prep(object):
    """A class to prepare input files for BICePs calculation"""

    def __init__(self,scheme=None,states=0.0,indices=None, exp_data=None, top=None, data_dir=None, out_dir=None):

        """Parameters
           ---------
        scheme: {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}
	states: number of states
        indices: experimental observable index (*.txt file)
        exp_data: experimental measuremnets (*.txt file)
        top: topology file (*.gro, pdb, etc.)
        data_dir: data directory (should have *txt file inside)
        out_dir: output directory
	"""
        if scheme not in ['noe','cs_H','cs_Ha','cs_N','cs_Ca','pf']:
            raise ValueError("scheme must be one of ['noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf']")
        else:
            if states==0.0 or indices == None or exp_data == None or top==None or data_dir==None:
                raise ValueError("make sure you have actual input for states, indices, exp_data, topology file or data directory ")
	

            if scheme in ['cs_H','cs_Ha','cs_Ca','cs_N']:
		self.write_cs_input()
	self.ind=np.loadtxt(indices)
	self.restraint_data = np.loadtxt(exp_data)
	self.topology = md.load(top).topology
	self.convert = lambda txt: int(txt) if txt.isdigit() else txt
	self.data = sorted(glob.glob(data_dir),key=lambda x: [self.convert(s) for s in re.split("([0-9]+)",x)])
        if out_dir == None:
            self.out = 'BICePs'+scheme
        self.out = out_dir
#	if self.ind.shape[0] != self.restraint_data.shape[0]:
#            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.')%(self.ind.shape[0],self.restraint_data.shape[0])



    def write_cs_input(self):
#        self.ind = np.loadtxt(indices)
#        self.restraint_data = np.loadtxt(exp_data)
        if self.ind.shape[0] != self.restraint_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.')%(self.ind.shape[0],self.restraint_data.shape[0])
#        self.topology = md.load(top).topology
        #convert = lambda txt: int(txt) if txt.isdigit() else txt
#        self.data = sorted(glob.glob(data_dir),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)])
	if int(len(self.data)) != int(states):
		raise ValueError("number of states doesn't equal to file numbers")
#	if out_dir == None:
#	    self.out = 'BICePs'+scheme
#	self.out = out_dir
	if not os.path.exists(self.out):
	    os.mkdir(self.out) 
        for j in xrange(len(self.data)):
	    model_data = np.loadtxt(self.data[j])
	    r = prep_cs()
	    all_atom_indices = [atom.index for atom in self.topology.atoms]
       	    all_atom_residues = [atom.residue for atom in self.topology.atoms]
       	    all_atom_names = [atom.name for atom in self.topology.atoms]
       	    for i in xrange(self.ind.shape[0]):
                a1 = int(self.ind[i])
                restraint_index = self.restraint_data[i,0]
                exp_chemical_shift        = self.restraint_data[i,1]
                model_chemical_shift      = model_data[i]
                r.add_line_cs(restraint_index, a1, self.topology, exp_chemical_shift, model_chemical_shift)
	    r.write('%s/%d.cs_H'%(self.out,j))

                 
                    

                
        

            
