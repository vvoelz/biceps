# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import mdtraj as md
import biceps.toolbox

class Preparation(object):

    def __init__(self, nstates=0,  top=None, outdir="./"):
        """A parent class to prepare input files for BICePs calculation.

        :param str obs: type of experimental observables {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}
        :param int default=0 nstates: number of states
        :param str default=None indices: experimental observable index (*.txt file)
        :param str default=None exp_data: experimental measuremnets (*.txt file)
        :param str default=None top: topology file (*.pdb)
        """

        self.nstates = nstates
        self.topology = md.load(top).topology
        self.data = list()

    def write_DataFrame(self, filename, to="pickle", verbose=True):
        """Write Pandas DataFrame to user specified filetype."""

        #biceps.toolbox.mkdir(self.outdir)
        #columns = { self.keys[i] : self.header[i] for i in range(len(self.keys)) }
        #print(columns)
        if verbose:
            print('Writing %s as %s...'%(filename,to))
        df = pd.DataFrame(self.biceps_df)
        #dfOut = getattr(self.df.rename(columns=columns), "to_%s"%to)
        dfOut = getattr(df, "to_%s"%to)
        dfOut(filename)

    #TODO: needs to be checked
    def prep_cs(self, exp_data, model_data, indices, extension, outdir=None):
        """A method containing input/output methods for writing chemicalshift
        Restaint Files.

        exp (ppm)
        model (ppm)
        """

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'exp', 'model', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i,0])
                dd['atom_index1'].append(a1)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            if verbose:
                print(self.biceps_df)
            filename = "%s.cs_%s"%(j, extension)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)



    def prep_noe(self, exp_data, model_data, indices, extension=None, outdir=None, verbose=False):
        """A method containing input/output methods for writing NOE
        Restaint Files.

        'exp' (A)
        'model' (A)
        """

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'atom_index2', 'res2', 'atom_name2', 'exp', 'model', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1, a2 = int(self.ind[i,0]), int(self.ind[i,1])
                dd['atom_index1'].append(a1)
                dd['atom_index2'].append(a2)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.noe"%(j)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)


    def prep_J(self, exp_data, model_data, indices, extension=None, outdir=None, verbose=False):
        """A method containing input/output methods for writing scalar coupling
        Restaint Files.

        'exp_J (Hz)
        'model_J (Hz)'
        """

        self.header = ('restraint_index', 'atom_index1', 'res1', 'atom_name1',
                'atom_index2', 'res2', 'atom_name2', 'atom_index3', 'res3', 'atom_name3',
                'atom_index4', 'res4', 'atom_name4', 'exp',
                'model', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        if type(indices) is not str:
            self.ind = indices
        else:
            self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray:
            self.model_data = biceps.toolbox.get_files(model_data)
        if int(len(self.model_data)) != int(self.nstates):
            raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1, a2, a3, a4   = int(self.ind[i,0]), int(self.ind[i,1]), int(self.ind[i,2]), int(self.ind[i,3])
                dd['atom_index1'].append(a1);dd['atom_index2'].append(a2)
                dd['atom_index3'].append(a3);dd['atom_index4'].append(a4)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['res2'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a2][0]))
                dd['atom_name2'].append(str([atom.name for atom in self.topology.atoms if atom.index == a2][0]))
                dd['res3'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a3][0]))
                dd['atom_name3'].append(str([atom.name for atom in self.topology.atoms if atom.index == a3][0]))
                dd['res4'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a4][0]))
                dd['atom_name4'].append(str([atom.name for atom in self.topology.atoms if atom.index == a4][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.J"%(j)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)


    # TODO: Needs to be checked
    def prep_pf(self, exp_data, model_data=None, indices=None, extension=None, outdir=None):
        """A method containing input/output methods for writing protection factor
        Restaint Files."""

        if model_data:
            self.header = ('restraint_index', 'atom_index1', 'res1', 'exp','model', 'comments')
        else:
            self.header = ('restraint_index', 'atom_index1', 'res1','exp', 'comments')
        self.exp_data = np.loadtxt(exp_data)
        self.model_data = model_data
        self.ind = np.loadtxt(indices, dtype=int)
        if type(self.model_data) is not list or np.ndarray or None:
            self.model_data = biceps.toolbox.get_files(model_data)
            if int(len(self.model_data)) != int(self.nstates):
                raise ValueError("The number of states doesn't equal to file numbers")
        if self.ind.shape[0] != self.exp_data.shape[0]:
            raise ValueError('The number of atom pairs (%d) does not match the\
                    number of restraints (%d)! Exiting.'%(self.ind.shape[0],self.exp_data.shape[0]))
        for j in range(len(self.model_data)):
            dd = { self.header[i]: [] for i in range(len(self.header)) }
            model_data = np.loadtxt(self.model_data[j])
            for i in range(self.ind.shape[0]):
                a1 = int(self.ind[i,0])
                dd['atom_index1'].append(a1)
                dd['res1'].append(str([atom.residue for atom in self.topology.atoms if atom.index == a1][0]))
                dd['atom_name1'].append(str([atom.name for atom in self.topology.atoms if atom.index == a1][0]))
                dd['restraint_index'].append(int(self.exp_data[i,0]))
                dd['exp'].append(np.float64(self.exp_data[i,1]))
                if model_data:
                    dd['model'].append(np.float64(model_data[i]))
                dd['comments'].append(np.NaN)
            self.biceps_df = pd.DataFrame(dd)
            if verbose:
                print(self.biceps_df)
            filename = "%s.pf"%(j)
            if outdir:
                self.write_DataFrame(filename=outdir+filename)






if __name__ == "__main__":

    import doctest
    doctest.testmod()





