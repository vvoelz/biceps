# -*- coding: utf-8 -*-
import os, sys, glob, string, re
import numpy as np
import pandas as pd
import mdtraj as md
sys.path.append("../")
from biceps.prep_cs import prep_cs
from biceps.prep_J import prep_J
from biceps.prep_noe import prep_noe
from biceps.prep_pf import prep_pf
import biceps.toolbox


class NMR_Chemicalshift(object):
    """A data containter class to store a datum for NMR chemical shift information."""

    def __init__(self, i, exp, model):
        """Initialize the derived NMR_Chemicalshift class.

        :param int i: atom indices from the conformation defining this chemical shift
        :var exp: the experimental chemical shift
        :var model: the model chemical shift in this structure (in ppm)

        >>> biceps.Observable.NMR_Chemicalshift(i, exp, model)
        """

        # Atom indices from the conformation defining this chemical shift
        self.i = i

        # the model chemical shift in this structure (in ppm)
        self.model = model

        # the experimental chemical shift
        self.exp = exp

        # N equivalent chemical shift should only get 1/N f the weight when
    #... computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0  #1.0/3.0 used in JCTC 2020 paper  # default is N=1



class NMR_Dihedral(object):
    """A data containter class to store a datum for NMR dihedral information."""

    def __init__(self, i, j, k, l, exp, model,
            equivalency_index=None, ambiguity_index=None):
        """Initialize NMR_Dihedral container class

        :param int i,j,k,l: atom indices from the conformation defining this dihedral
        :var exp: the experimental J-coupling constant
        :var model:  the model distance in this structure (in Angstroms)
        :var equivalency_index: the index of the equivalency group (i.e. a tag for equivalent H's)
        :var ambiguity_index: the index of the ambiguity group \n (i.e. some groups distances\
                have distant values, but ambiguous assignments.  Posterior sampling can be performed over these values)

        >>> biceps.Observable.NMR_Dihedral(i, j, k, l, exp,  model)
        """

        # Atom indices from the conformation defining this dihedral
        self.i = i
        self.j = j
        self.k = k
        self.l = l

        # the model distance in this structure (in Angstroms)
        self.model = model

        # the experimental J-coupling constant
        self.exp = exp

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent distances should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1

        # the index of the ambiguity group (i.e. some groups distances have
        # distant values, but ambiguous assignments.  We can do posterior sampling over these)
        self.ambiguity_index = ambiguity_index


class NMR_Distance(object):
    """A class to store NMR noe information."""

    def __init__(self, i, j, exp, model, equivalency_index=None):
        """Initialize NMR_Distance container class

        :param int i,j: atom indices from the conformation defining this noe
        :var exp: the experimental NOE noe (in Angstroms)
        :var model: the model noe in this structure (in Angstroms)
        :var equivalency_index: the index of the equivalency group (i.e. a tag for equivalent H's)

        >>> biceps.Observable.NMR_Distance(i, j, exp,  model)
        """

        # Atom indices from the conformation defining this noe
        self.i = i
        self.j = j

        # the model noe in this structure (in Angstroms)
        self.model = model

        # the experimental NOE noe (in Angstroms)
        self.exp = exp

        # the index of the equivalency group (i.e. a tag for equivalent H's)
        self.equivalency_index = equivalency_index

        # N equivalent noe should only get 1/N of the weight when computing chi^2
        self.weight = 1.0  # default is N=1


class NMR_Protectionfactor(object):
    """A class to store NMR protection factor information."""

    def __init__(self, i, exp, model):
        """Initialize NMR_Protectionfactor container class

        :param int i: atom indices from the conformation defining this protection factor
        :var exp: the experimental protection factor
        :var model: the model protection factor in this structure


        >>> biceps.Observable.NMR_Protectionfactor(i, exp,  model)
        """

        # Atom indices from the conformation defining this protection factor
        self.i = i

        # the model protection factor in this structure (in ???)
        self.model = model

        # the experimental protection factor
        self.exp = exp

        # N equivalent protection factor should only get 1/N f the weight when computing chi^2 (not likely in this case but just in case we need it in the future)
        self.weight = 1.0 # default is N=1




class Preparation(object):

    def __init__(self, nstates=0, indices=None, top=None, outdir="./", precomputed=False):
        """A parent class to prepare input files for BICePs calculation.

        :param str obs: type of experimental observables {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}
        :param int default=0 nstates: number of states
        :param str default=None indices: experimental observable index (*.txt file)
        :param str default=None exp_data: experimental measuremnets (*.txt file)
        :param str default=None top: topology file (*.pdb)
        :param str default=None model_data: precomputed data directory (should have *txt file inside)
        """

        self.nstates = nstates
        self.ind = np.loadtxt(indices, dtype=int)
        self.topology = md.load(top).topology
        self.data = list()
        #self.res_names  = [atom.residue for atom in self.topology.atoms if atom.index == i][0]
        #self.atom_names = [atom.name for atom in self.topology.atoms if atom.index == i][0]
        #self.atom_indices = [atom.index for atom in self.topology.atoms]

    def write_DataFrame(self, filename, to="pickle", verbose=True):
        """Write Pandas DataFrame to user specified filetype."""

        #biceps.toolbox.mkdir(self.outdir)
        #columns = { self.keys[i] : self.header[i] for i in range(len(self.keys)) }
        #print(columns)
        if verbose:
            print('Writing %s as %s...'%(filename,to))
        self.df = pd.DataFrame(self.biceps_df)
        #dfOut = getattr(self.df.rename(columns=columns), "to_%s"%to)
        dfOut = getattr(self.df, "to_%s"%to)
        dfOut(self.outdir+filename)


    def prep_noe(self, exp_data, model_data, extension, outdir=None):
        pass
    def prep_J(self, exp_data, model_data, extension, outdir=None):
        pass
    def prep_pf(self, exp_data, model_data, extension, outdir=None):
        pass

    '''
    def write(self, outdir="./"):
        """Write BICePs format input files for available experimental observables.

        :param str default=None out_dir: output directory
        """

        if self.obs not in biceps.toolbox.list_possible_extensions():
            raise ValueError(f"obs must be one of {biceps.toolbox.list_possible_extensions()}")
        else:
            self.out = outdir
        biceps.toolbox.mkdir(self.out)
        for j in range(len(self.data)):
            model_data = np.loadtxt(self.data[j])

            # Find all Child Restraint classes in the current file
            current_module = sys.modules[__name__]
            # Pick the Restraint class upon file extension
            prep_obs = getattr(current_module, "prep_%s"%(obs))

            # Get all the arguments for the Child Restraint Class
            args = {"%s"%key: val for key,val in locals().items()
                    if key in inspect.getfullargspec(prep_obs)[0]
                    if key != 'self'}
            print(f"args = {args}")
            print(f"Required args by inspect:{inspect.getfullargspec(prep_obs)[0]}")
            exit()
            prep_obs(**args)


    def write_noe_input(self):
        for j in range(len(self.data)):
            model_data = np.loadtxt(self.data[j])
            r = prep_noe()
            all_atom_indices = [atom.index for atom in self.topology.atoms]
            all_atom_residues = [atom.residue for atom in self.topology.atoms]
            all_atom_names = [atom.name for atom in self.topology.atoms]
            for i in range(self.ind.shape[0]):
                a1, a2 = int(self.ind[i,0]), int(self.ind[i,1])
                restraint_index = self.exp_data[i,0]
                exp_noe        = self.exp_data[i,1]
                model_noe      = model_data[i]
                r.add_line(restraint_index, a1, a2, self.topology, exp_noe, model_noe)
            r.write('%s/%d.%s'%(self.out,j,self.obs))

    def write_J_input(self):
        for j in range(len(self.data)):
            model_data = np.loadtxt(self.data[j])
            r = prep_J()
            all_atom_indices = [atom.index for atom in self.topology.atoms]
            all_atom_residues = [atom.residue for atom in self.topology.atoms]
            all_atom_names = [atom.name for atom in self.topology.atoms]
            for i in range(self.ind.shape[0]):
                a1, a2, a3, a4 = int(self.ind[i,0]), int(self.ind[i,1]), int(self.ind[i,2]), int(self.ind[i,3])
                restraint_index = self.exp_data[i,0]
                exp_J_coupling      = self.exp_data[i,1]
                model_J_coupling      = model_data[i]
                r.add_line(restraint_index, a1, a2, a3, a4, self.topology, exp_J_coupling, model_J_coupling)
#            r.add_line(restraint_index, a1, a2, a3, a4, self.topology, J_coupling, self.karplus)
            r.write('%s/%d.%s'%(self.out,j,self.obs))

    def write_pf_input(self):
        if precomputed:
            for j in range(len(self.data)):
                model_data = np.loadtxt(self.data[j])
                r = prep_pf()
                all_atom_indices = [atom.index for atom in self.topology.atoms]
                all_atom_residues = [atom.residue for atom in self.topology.atoms]
                all_atom_names = [atom.name for atom in self.topology.atoms]
                for i in range(self.ind.shape[0]):
                    a1 = int(self.ind[i])
                    restraint_index = self.exp_data[i,0]
                    exp_pf          = self.exp_data[i,1]
                    protectionfactor        = model_data[i]
                    r.add_line(restraint_index, a1, self.topology, exp_pf, protectionfactor)
                r.write('%s/%d.%s'%(self.out,j,self.obs))
        else:
            for j in range(len(self.data)):
                r = prep_pf()
                all_atom_indices = [atom.index for atom in self.topology.atoms]
                all_atom_residues = [atom.residue for atom in self.topology.atoms]
                all_atom_names = [atom.name for atom in self.topology.atoms]
                for i in range(self.ind.shape[0]):
                    a1 = int(self.ind[i])
                    restraint_index = self.exp_data[i,0]
                    exp_pf          = self.exp_data[i,1]
                    r.add_line(restraint_index, a1, self.topology, exp_pf)
                r.write('%s/%d.%s'%(self.out,j,self.obs))

    '''



    def prep_cs(self, exp_data, model_data, extension, outdir=None):
        """A method containing input/output methods for writing chemicalshift
        Restaint Files."""

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'exp_chemical_shift (ppm)', 'model_chemical_shift (ppm)', 'comments')

        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.')%(self.ind.shape[0],self.exp_data.shape[0])

        ## Removing units to have simplified dictionary keys()
        #self.keys = [ self.header[i].split("(")[0].strip() for i in range(len(self.header)) ]
        self.data_dict = { self.header[i]: [] for i in range(len(self.header)) }
        self.biceps_df = pd.DataFrame(self.data_dict)
        for j in range(len(self.model_data)):
            for i in range(self.ind.shape[0]):
                resname1  = [atom.residue for atom in self.topology.atoms if atom.index == i][0]
                atomname1 = [atom.name for atom in self.topology.atoms if atom.index == i][0]
                a1                  = int(self.ind[i])
                restraint_index     = self.exp_data[i,0]
                exp_chemical_shift  = self.exp_data[i,1]
                model_chemical_shift= model_data[i]
                df = pd.DataFrame({
                    'restraint_index': restraint_index, 'atom_index1': a1,
                    'res1': resname1, 'atom_name1': atomname1,
                    'exp_chemical_shift (ppm)': exp_chemical_shift,
                    'model_chemical_shift (ppm)': model_chemical_shift,
                    'comments': None # TODO: ?
                        })
                self.biceps_df.append(df)
            filename = "%s.cs_%s"%(j, extension)
            if outdir:
                self.outdir = outdir
                self.write_DataFrame(filename=outdir+filename)






if __name__ == "__main__":

    #import doctest
    #doctest.testmod()

    prep_cs("test")




