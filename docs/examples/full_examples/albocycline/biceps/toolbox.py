# -*- coding: utf-8 -*-
import sys, os, glob
import numpy as np
import re
import yaml, io
from biceps.J_coupling import *
from biceps.KarplusRelation import *
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def sort_data(dataFiles):
    """Sorting the data by extension into lists. Data can be located in various
    directories.  Provide a list of paths where the data can be found.
    Some examples of fileextensions: {.noe,.J,.cs_H,.cs_Ha}.

    :param list dataFiles: list of strings where the data can be found
    :raises ValueError: if the data directory does not exist
    """

    dir_list=[]
    if not os.path.exists(dataFiles):
        raise ValueError("data directory doesn't exist")
    if ',' in dataFiles:
        print('Sorting out the data...\n')
        raw_dir = (dataFiles).split(',')
        for dirt in raw_dir:
            if dirt[-1] == '/':
                dir_list.append(dirt+'*')
            else:
                dir_list.append(dirt+'/*')
    else:
        raw_dir = dataFiles
        if raw_dir[-1] == '/':
            dir_list.append(dataFiles+'*')
        else:
            dir_list.append(dataFiles+'/*')

    data = [[] for x in range(7)] # list for every extension; 7 possible experimental observables supported
    # Sorting the data by extension into lists. Various directories is not an issue...
    for i in range(0,len(dir_list)):
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        # This convert / sorted glob is a bit fishy... needs many tests
        for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
            if j.endswith('.cs_H'):
                data[0].append(j)
            elif j.endswith('.cs_Ha'):
                data[1].append(j)
            elif j.endswith('.cs_N'):
                data[2].append(j)
            elif j.endswith('.cs_Ca'):
                data[3].append(j)
            elif j.endswith('.J'):
                data[4].append(j)
            elif j.endswith('.pf'):
                data[5].append(j)
            elif j.endswith('.noe'):
                data[6].append(j)

            else:
                raise ValueError("Incompatible file extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
    data = np.array([_f for _f in data if _f]) # removing any empty lists
    Data = np.stack(data, axis=-1)
    data = Data.tolist()
    return data


def list_res(input_data):
    """Determine what scheme is included in sampling"""

#    input_data = sort_data(data)
    scheme=[]
    for i in input_data[0]:
        if i.endswith('.cs_H'):
            scheme.append('cs_H')
        elif i.endswith('.cs_Ha'):
            scheme.append('cs_Ha')
        elif i.endswith('.cs_N'):
            scheme.append('cs_N')
        elif i.endswith('.cs_Ca'):
            scheme.append('cs_Ca')
        elif i.endswith('.J'):
            scheme.append('J')
        elif i.endswith('.pf'):
            scheme.append('pf')
        elif i.endswith('.noe'):
            scheme.append('noe')
#                scheme.append('gamma')
        else:
            raise ValueError("Incompatible File extension. Use:{*.noe, *.J, *.cs_H, *.cs_Ha, *.cs_N, *.cs_Ca, *.pf}")

    return scheme


def mkdir(path):
    """Create directory if path does not exist.

    >>> toolbox.mkdir("./doctest")
    """

    # create a directory for each system
    if not os.path.exists(path):
        os.mkdir(path)


def write_results(self, outfilename):
    """Writes a compact file of several arrays into binary format.
    Standardized: Yes ; Binary: Yes; Human Readable: No;

    :param str outfilename: name of the output file
    :return: numpy compressed filetype
    """

    np.savez_compressed(outfilename, self.results)

def read_results(self,filename):
    """Reads a numpy compressed filetype(*.npz) file"""

    loaded = np.load(filename)
    print((list(loaded.items())))

def convert_pop_to_energy(pop_filename, out_filename=None):
    """Convert population to energy for each state using the following:

      >>> U = -np.log(P)

    :param str pop_filename: name of file for populations
    :param str out_filename: output file name
    :return list: A list of converted energy for each conformational state
    """

    if pop_filename.endswith('txt') or pop_filename.endswith('dat'):
        pop = np.loadtxt(pop_filename)
    elif pop_filename.endswith('npy'):
        pop = np.load(pop_filename)
    else:
        raise ValueError('Incompatible file extention. Use:{.txt,.dat,.npy}')
    energy=[]
    # replace NaN in the list with a very small number
    pop[np.isnan(pop)]=0.001
    for i in pop:
        energy.append(-np.log((i/float(sum(pop)))))

    if out_filename == None:
        np.savetxt('energy.txt',energy)
    else:
        np.savetxt(out_filename,energy)

    return energy

def get_J3_HN_HA(top,traj=None, frame=None,  model="Habeck", outname = None):
    '''Compute J3_HN_HA for frames in a trajectories.

    :param mdtraj.Trajectory traj: Trajectory
    :param mdtraj.Topology top: topology file
    :param list frame: specific frame for computing
    :param str model: Karplus coefficient models
      ["Ruterjans1999","Bax2007","Bax1997","Habeck" ,"Vuister","Pardi"]
    :param str outname: if not None, the output will be saved and a
      file name (in the format of string) is required.'''

    J=[]
    if traj is not None:
        if frame is None:
            t = md.load(traj,top=top)
            J = compute_J3_HN_HA(t, model = model)
        elif frame is not None:
            for i in range(len(frame)):
                t = md.load(traj,top=top)[frame[i]]
                d = compute_J3_HN_HA(t, model = model)
                if i == 0:
                    J.append(d[0])
                    J.append(d[1])
                else:
                    J.append(d[1])
    else:
        t = md.load(top)
        J = compute_J3_HN_HA(t, model = model)
    if outname is not None:
        print('saving output file...')
        np.save(outname, J)
        print('Done!')
    else:
        print('saving output file ...')
        np.save('J3_coupling',J)
        print('Done!')
    return J

def dihedral_angle(x0, x1, x2, x3):
    """Calculate the signed dihedral angle between 4 positions. Result is in degrees.

    :param float x0:
    :param float x1:
    :param float x2:
    :param float x3:
    :return float phi: dihedral angle in degrees
    """

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

def compute_nonaa_Jcoupling(traj, index, karplus_key, top=None):
    """Compute J couplings for small molecules.

    :param mdtraj.Trajectory traj: Trajectory or *.pdb/*.gro files
    :param index: index file for atoms
    :param karplus_key: karplus relation for each J coupling
    :param mdtraj.Topology default=None top: topology file (only required if a trajectory is loaded)"""


    if len(karplus_key) != len(index):
        raise ValueError("The number of index must equale the number of karplus_key.")
    if traj.endswith('.gro'):
        conf = md.load(traj)
    elif traj.endswith('.pdb'):
        conf = md.load(traj)
    else:
        if top == None:
            raise TypeError("To load a trajectory file, a topology file must be provided.")
        conf = md.load(traj,top=top)
    J = np.zeros((len(conf),len(index)))
    karplus = KarplusRelation()
    for i in range(len(J)):
        for j in range(len(index)):
            ri, rj, rk, rl = [conf.xyz[0,x,:] for x in index[j]]
            model_angle = dihedral_angle(ri, rj, rk, rl)
            J[i,j] = karplus.J(model_angle, karplus_key[j])
    return J

def plot_ref(traj, debug = True):
    #from matplotlib import pyplot as plt
    # Load in yaml trajectories
    #output = os.path.join(resultdir,'traj_lambda0.00.npz')
    #output = traj
    """Plot reference potential for each observables.

    :param traj: output trajectory from BICePs sampling
    :return figure: A figure of reference potential and distribution of model observables
    """

    if debug:
        print('Loading %s ...'%traj)
    results = np.load(traj)['arr_0'].item()
    n_restraints = len(results['ref_potential'])
    for i in range(n_restraints):
        if results['ref_potential'][i][0] == 'Nan':
            pass
        else:
            n_model = len(results['ref_potential'][i][0])
            c,r = 5, int(n_model)/5 + 1
            x = np.arange(0.0,30.0,0.01)
            print("plotting figures...")
            plt.figure(figsize=(4*c,5*r))
            if len(results['ref_potential'][i]) == 1:   ## exp ##
                for j in range(n_model):
                    beta = results['ref_potential'][i][0][j]
                    model = results['model'][i][j]
                    ref = np.exp(-x/beta)/beta
                    counts,bins = np.histogram(model,bins = np.arange(0.0,20.0,0.2))
                    plt.subplot(r,c,j+1)
                    plt.step(bins[0:-1],counts,'black',label = '$P^{d_j}$')
                    plt.plot(x,ref*10.,'blue',label='$P_{ref}(d_j)$')
                    plt.xlim(0.0,max(model))
                    plt.yticks([])
                    plt.legend(loc='upper right',fontsize=8)
                plt.tight_layout()
                plt.savefig('ref_distribution.pdf')
                plt.close()
            elif len(results['ref_potential'][i]) == 2:   ## gau ##
                for j in range(n_model):
                    mean = results['ref_potential'][i][0][j]
                    sigma = results['ref_potential'][i][1][j]
                    model = results['model'][i][j]
                    ref = (1.0/(np.sqrt(2.0*np.pi*sigma**2.0)))*np.exp(-(x-mean)**2.0/(2.0*sigma**2.0))
                    counts,bins = np.histogram(model,bins = np.arange(0.0,20.0,0.2))
                    plt.subplot(r,c,j+1)
                    plt.step(bins[0:-1],counts,'black',label = '$P^{d_j}$')
                    plt.plot(x,ref*10.,'blue',label='$P_{ref}(d_j)$')
                    plt.xlim(0.0,max(model))
                    plt.yticks([])
                    plt.legend(loc='upper right',fontsize=8)
                plt.tight_layout()
                plt.savefig('ref_distribution.pdf')
                plt.close()
                print("Done!")



def get_rest_type(traj):
    """Get types of experimental restraints.

    :param traj: output trajectory from BICePs sampling
    :return list: A list of types of experimental restraints
    """

    rest_type=[]
    if not traj.endswith('.npz'):
        raise TypeError("trajectory file should be in the format of '*npz'")
    else:
        t = np.load(traj)['arr_0'].item()
        rest = t['rest_type']
        for r in rest:
            if r.split('_')[1] != 'noe':
                rest_type.append(r.split('_')[1])
            elif r.split('_')[1] == 'noe':
                rest_type.append(r.split('_')[1])
                rest_type.append('gamma')
    return rest_type



def plot_grid(traj, rest_type=None):
    """Plot acceptance ratio for each nuisance parameters jump during MCMC sampling.
    """
    if rest_type == None:
        rest = get_rest_type(traj)
    else:
        rest = rest_type
    t = np.load(traj)['arr_0'].item()
    grid = t['grid']
    for i in range(len(grid)):
        plt.figure()
        cmap=plt.get_cmap('Greys')
        raw = grid[i]
        max_n = np.max(raw)
        raw[raw == 0.] = -1.0*max_n
        plt.pcolor(raw,cmap=cmap,vmin=-1.0*max_n,vmax=max_n,edgecolors='none')
        plt.colorbar()
        if rest[i] == "gamma":
            plt.xlabel('$\gamma$ $index$')
            plt.ylabel('$\gamma$ $index$')
        else:
            plt.xlabel('$\sigma_{%s}$ $index$'%rest[i])
            plt.ylabel('$\sigma_{%s}$ $index$'%rest[i])
        plt.xlim(0,len(raw[i]))
        plt.ylim(0,len(raw[i]))
        plt.savefig('grid_%s.pdf'%rest[i])
        plt.close()


def find_all_state_sampled_time(trace,nstates):
    frac = []
    all_states = np.zeros(nstates)
    init = 0
    while 0 in all_states:
        if init == len(trace):
            print('not all state sampled, these states', np.where(all_states == 0)[0],'are not sampled')
            return 'null', frac
        else:
        #    print trace[init]
            all_states[trace[init]] += 1
            frac.append(float(len(np.where(all_states!=0)[0]))/float(nstates))
            init += 1
    return init, frac



if __name__ == "__main__":

    import doctest
    doctest.testmod()









