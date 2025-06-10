# -*- coding: utf-8 -*-
import os, glob, re, pickle, inspect
import h5py
import numpy as np
import pandas as pd
import psutil, gc, sys, time

import biceps
from biceps.J_coupling import *
from biceps.KarplusRelation import KarplusRelation

import mdtraj as md
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import biceps.Restraint as Restraint
from scipy.optimize import curve_fit
import Bio
from Bio import SeqUtils
from natsort import natsorted


mpl_colors = matplotlib.colors.get_named_colors_mapping()
mpl_colors = list(mpl_colors.values())[::5]
extra_colors = mpl_colors.copy()
#mpl_colors = ["k","lime","b","brown","c","green",
mpl_colors = ["b","brown","c","green",
              "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
              #'#acc2d9', "orange", '#894585', '#fcfc81', '#efb435', '#3778bf',
        '#e78ea5', '#983fb2', '#b7e1a1', '#430541', '#507b9c', '#c9d179',
            '#2cfa1f', '#fd8d49', '#b75203', '#b1fc99']+extra_colors[::2]+ ["k","grey"]



def get_PSkwargs(locals={}, exclude=None):
    PSkwargs = inspect.getfullargspec(biceps.PosteriorSampler)
    exclude = [] if exclude is None else exclude
    kwargs = {key: val for key, val in locals.items() if key in PSkwargs.args and key != 'self' and key not in exclude}
    kwargs.update({key: val for key, val in zip(PSkwargs.args[-len(PSkwargs.defaults):], PSkwargs.defaults) if key not in kwargs})
    return kwargs

def get_sample_kwargs(locals={}):
    obj = getattr(biceps.PosteriorSampler, "sample")
    sample_kwargs = inspect.getfullargspec(obj)
    kwargs = {key: val for key, val in locals.items() if key in sample_kwargs.args and key != 'self'}
    kwargs.update({key: val for key, val in zip(sample_kwargs.args[-len(sample_kwargs.defaults):], sample_kwargs.defaults) if key not in kwargs})
    return kwargs


def three2one(string):
    match = re.match(r"([a-z]+)([0-9]+)", string[0].upper()+string[1:].lower(), re.I)
    items = match.groups()
    code = SeqUtils.IUPACData.protein_letters_3to1[items[0]]
    return code+items[-1]

def one2three(string):
    match = re.match(r"([a-z]+)([0-9]+)", string[0].upper()+string[1:].lower(), re.I)
    items = match.groups()
    code = SeqUtils.IUPACData.protein_letters_1to3[items[0]]
    return code+items[-1]





def get_forward_model(ensemble):
    """Returns the model with all model data"""

    model = []
    for s in range(len(ensemble)):
        _model = []
        for R in ensemble[s]:
            m = []
            for j in range(R.n):
                m.append(R.restraints[j]["model"])
            _model.append(m)
        model.append(_model)
    return model


def format_label(label):

    if label.count('_') == 1:
        label = "$\%s_{%s}$"%(label.split('_')[0],label.split('_')[1])
    elif label.count('_') == 2:
        label = "$\%s_{{%s}_{%s}}$"%(label.split('_')[0],label.split('_')[1],label.split('_')[2])

    if 'gamma' in label: label = label.replace("\gamma", "{\gamma}^{-1/6}")
    return label






def ngl_align_CLN001(structure_files, ref=0, gui=True, rep='cartoon', show_resids=None, with_distance=False):
    """
    #https://docs.mdanalysis.org/stable/documentation_pages/analysis/align.html#MDAnalysis.analysis.align.alignto
    #http://nglviewer.org/nglview/latest/api.html#nglview.write_html
    """

    import nglview as ngl
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    mpl_colors = matplotlib.colors.cnames.keys()
    mpl_colors = list(mpl_colors)[::5]

    _repr_ = [
            {"type": rep, "params": {
                "color": "random",
                "sele": "all",
        }}
    ]
    frame = mda.Universe(structure_files[ref])
    t = ngl.MDAnalysisTrajectory(frame)
    w = ngl.NGLWidget(t, gui=True)
    w.set_representations(_repr_, component=0)

    if with_distance: w.add_distance(atom_pair=[["3.N", "7.O"]], label_color="black")
    if with_distance: w.add_distance(atom_pair=[["3.N", "8.OG1"]], label_color="black")
    if show_resids:
        for id in show_resids:
            w.add_ball_and_stick(f'{id}:A')

    _structure_files = structure_files.copy()
    _structure_files.pop(int(ref))
    ref = mda.Universe(structure_files[ref])
    for i,file in enumerate(_structure_files):
        struct = mda.Universe(file)
        rmsds = align.alignto(struct, ref,
                #select='name CA', # selection to operate on
                select='all', # selection to operate on
                match_atoms=True) # whether to match atoms
        t = mda.Merge(struct.atoms, ref.atoms)
        s = ngl.MDAnalysisTrajectory(t)
        w.add_structure(s, align=True)
        _repr_[0]["params"]["color"] = mpl_colors[i]
        w.set_representations(_repr_, component=int(i+1))
        w.update_representation(component=int(i+1), repr_index=0, **_repr_[0]["params"])
        if with_distance: w.add_distance(atom_pair=[["3.N", "7.O"]], label_color="black", component=int(i+1))
        if with_distance: w.add_distance(atom_pair=[["3.N", "8.OG1"]], label_color="black", component=int(i+1))
        if show_resids:
            for id in show_resids:
                w.add_ball_and_stick(f'{id}:A', component=int(i+1))

    return w



def ngl_align(structure_files, weights=None, ref=0, gui=True, alignto="all",
              rep='cartoon', show_resids=None, print_rmsds=False,
              add_distances=[]):
    """
    Aligns and displays structures with transparency levels based on provided weights.
    Higher weights result in less transparency.
    """

    import nglview as ngl
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    mpl_colors = matplotlib.colors.cnames.keys()
    mpl_colors = list(mpl_colors)[::5]


    if weights is None:
        weights = [1] * len(structure_files)  # Default equal weight if none provided
    max_weight = max(weights)

    mpl_colors = list(matplotlib.colors.cnames.keys())[::5]

    frame = mda.Universe(structure_files[ref])
    t = ngl.MDAnalysisTrajectory(frame)
    w = ngl.NGLWidget(t, gui=gui)

    _repr_ = [{
        "type": rep, "params": {
            "color": mpl_colors[0 % len(mpl_colors)],
            "sele": "all",
            "opacity": weights[0] / max_weight
        }
    }]
    w.set_representations(_repr_, component=0)

    if show_resids:
        for id in show_resids:
            w.add_ball_and_stick(f'{id}:A')

    ref_structure = mda.Universe(structure_files[ref])
    for i, file in enumerate(structure_files):
        if i == ref:
            continue  # Skip the reference file as it is already loaded
        struct = mda.Universe(file)
        rmsds = align.alignto(struct, ref_structure, select=alignto, match_atoms=True)
        if print_rmsds:
            print("%s: %0.2fÅ" % (file.split("/")[-1], rmsds[1]))

        t = mda.Merge(struct.atoms, ref_structure.atoms)
        s = ngl.MDAnalysisTrajectory(t)
        w.add_structure(s, align=True)
        _repr_[0]["params"]["color"] = mpl_colors[i % len(mpl_colors)]
        _repr_[0]["params"]["opacity"] = weights[i] / max_weight  # Normalize opacity to the max weight
        w.set_representations(_repr_, component=int(i+1))

        if show_resids:
            for id in show_resids:
                w.add_ball_and_stick(f'{id}:A', component=int(i+1))

    return w

#
#def ngl_align(structure_files, ref=0, gui=True, alignto="all",
#              rep='cartoon', show_resids=None, print_rmsds=False,
#              add_distances=[]):
#    """
#    #https://docs.mdanalysis.org/stable/documentation_pages/analysis/align.html#MDAnalysis.analysis.align.alignto
#    #http://nglviewer.org/nglview/latest/api.html#nglview.write_html
#    """
#
#    import nglview as ngl
#    import MDAnalysis as mda
#    from MDAnalysis.analysis import align
#
#    #mpl_colors = matplotlib.colors.get_named_colors_mapping()
#    #mpl_colors = list(mpl_colors.values())[::5]
#    mpl_colors = matplotlib.colors.cnames.keys()
#    mpl_colors = list(mpl_colors)[::5]
#
#    _repr_ = [
#            {"type": rep, "params": {
#                "color": "random",
#                "sele": "all",
#        }}
#    ]
#    frame = mda.Universe(structure_files[ref])
#    t = ngl.MDAnalysisTrajectory(frame)
#    w = ngl.NGLWidget(t, gui=True)
#    w.set_representations(_repr_, component=0)
#
#    if show_resids:
#        for id in show_resids:
#            w.add_ball_and_stick(f'{id}:A')
#
#    _structure_files = structure_files.copy()
#    _structure_files.pop(int(ref))
#    ref = mda.Universe(structure_files[ref])
#    for i,file in enumerate(_structure_files):
#        struct = mda.Universe(file)
#        rmsds = align.alignto(struct, ref,
#                select=alignto, #'name CA', # selection to operate on
#                #select='all', # selection to operate on
#                match_atoms=True) # whether to match atoms
#        if print_rmsds: print("%s: %0.2fÅ"%(file.split("/")[-1], rmsds[1]))
#        t = mda.Merge(struct.atoms, ref.atoms)
#        s = ngl.MDAnalysisTrajectory(t)
#        w.add_structure(s, align=True)
#        if i == 0:
#            _repr_[0]["params"]["color"] = "red"
#        else:
#            _repr_[0]["params"]["color"] = mpl_colors[i]
#
#        #_repr_[0]["params"]["repr"] = rep
#        #w.update_representation(component=int(i+1), repr_index=0, color=mpl_colors[i])
#        w.set_representations(_repr_, component=int(i+1))
#        w.update_representation(component=int(i+1), repr_index=0, **_repr_[0]["params"])
##        if add_distances != []:
##            for pair in add_distances:
##                w.add_distance(atom_pair=[pair], label_color="black", component=int(i+1))
#        #w.add_distance(atom_pair=[["3.N", "8.OG1"]], label_color="black", component=int(i+1))
#        if show_resids:
#            for id in show_resids:
#                w.add_ball_and_stick(f'{id}:A', component=int(i+1))
#
#
#
#        #w.set_representations(_repr_, component=int(i+1))
#        #component = getattr(w, f"component_{i+1}")
#        #component.set_representations(_repr_, component=int(i+1))
#
#        #w.add_representation(rep, selection='*')
#    return w




def align_and_save_pdbs(pdb_files, output_file):
    # Load the first pdb file to use as a reference
    reference = md.load(pdb_files[0])

    with open(output_file, 'w') as output:
        for pdb_file in pdb_files:
            # Load and align the pdb file
            traj = md.load(pdb_file)
            traj.superpose(reference)

            # Save the aligned trajectory to a temporary file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as tmp_file:
                traj.save(tmp_file.name)

                # Go back to the start of the temporary file
                tmp_file.seek(0)

                # Insert a comment line and write the contents of the temporary file
                output.write(f"REMARK Original file: {pdb_file}\n")
                for line in tmp_file:
                    if not line.startswith("CRYST1") and not line.startswith("MODEL") and not line.startswith("END"):
                        output.write(line)
                output.write("END\n")

            # Remove the temporary file
            os.remove(tmp_file.name)



def get_rmsds(structure_files, ref=0, alignto="all"):
    """
    #https://docs.mdanalysis.org/stable/documentation_pages/analysis/align.html#MDAnalysis.analysis.align.alignto
    #http://nglviewer.org/nglview/latest/api.html#nglview.write_html
    """

    import nglview as ngl
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    frame = mda.Universe(structure_files[ref])
    t = ngl.MDAnalysisTrajectory(frame)
    _structure_files = structure_files.copy()
    if not isinstance(_structure_files, list):
        _structure_files = list(_structure_files)
    _structure_files.pop(int(ref))
    ref = mda.Universe(structure_files[ref])
    result = []
    for i,file in enumerate(_structure_files):
        struct = mda.Universe(file)
        rmsds = align.alignto(struct, ref,
                select=alignto, #'name CA', # selection to operate on
                match_atoms=True) # whether to match atoms
        result.append(rmsds[1])
    return result







###############################################################################
#TODO: Does sort_data need to be so elaborate? Can't we just sort by order in Restraint.py?
###############################################################################
def sort_data(dataFiles):
    """Sorting the data by extension into lists. Data can be located in various
    directories.  Provide a list of paths where the data can be found.
    Some examples of file extensions: {.noe,.J,.cs_H,.cs_Ha}.

    :param list dataFiles: list of strings where the data can be found
    :raises ValueError: if the data directory does not exist

    >>> biceps.toolbox.sort_data()
    """

    dir_list=[]
    path, ext = os.path.splitext(dataFiles)
    if path.endswith("*"): path = path.replace("*","")

    if not os.path.exists(path):
        raise ValueError("data directory doesn't exist")

    raw_dir = dataFiles

    # Check if raw_dir ends with os.sep or contains a wildcard (*)
    if raw_dir.endswith(os.sep) or "*" in raw_dir:
        dir_path = raw_dir if "*" in raw_dir else os.path.join(dataFiles, '*')
    else:
        # If raw_dir is a file name with extension, assume all files of this type are needed
        _, ext = os.path.splitext(raw_dir)
        dir_path = os.path.join(dataFiles, '*' + ext) if "*" in ext else os.path.join(dataFiles, '*')
    dir_list.append(dir_path)
    files = get_files(dir_path)


#    if raw_dir.endswith(os.sep):
#        dir_list.append(os.path.join(dataFiles, '*'))
#        files = get_files(os.path.join(dataFiles, '*'))
#    elif "*" in os.path.splitext(raw_dir)[-1]:
#        dir_list.append(raw_dir)
#        files = get_files(dataFiles)
#    else:
#        _, ext = os.path.split(raw_dir)
#        if "*" in ext:
#            dir_list.append(dataFiles)
#            files = get_files(dataFiles)
#        else:
#            dir_list.append(os.path.join(dataFiles, '*'))
#            files = get_files(os.path.join(dataFiles, '*'))


    types = list_extensions(files)
    #print(types)
    data = [[] for t in types]
    #print(data)

    # Sorting the data by extension into lists. Various directories is not an issue...
    for i in range(len(dir_list)):
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        # This convert / sorted glob is a bit fishy... needs many tests
        for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
            name, ext = os.path.splitext(j)
            if not any([ext.startswith(possible) for possible in list_possible_extensions()]):
                raise ValueError(f"Incompatible File extension. You gave: {ext}; Extension should start with:{list_possible_extensions()}")
            else:
                for k,obs_type in enumerate(types):
                    #if ext.startswith(obs_type):
                    if ext.endswith(obs_type):
                        data[k].append(j)
    data = np.array([_f for _f in data if _f]) # removing any empty lists
    Data = np.stack(data, axis=-1)
    data = Data.tolist()
    return data



def get_files(path):
    """Return a sorted list of files that will be globbed from the path given.
    First, this function can handle decimals and multiple numbers that are
    seperated by characters.
    https://pypi.org/project/natsort/

    Args:
        path(str) - path that will be globbed
    Returns:
        sorted list
    """
#    path = path.replace("[", r"\[").replace("]", r"\]")
#    path = path.replace('[', '[[]').replace(']', '[]]')
    globbed = glob.glob(path)
    return natsorted(globbed)








def list_res(input_data):
    """Determine what scheme is included in sampling

    >>> biceps.toolbox.list_res()
    """
    scheme = [t[1:] for t in list_extensions(input_data)]
    return scheme

def list_extensions(input_data):
    if not isinstance(input_data, list): ValueError("input_data is not a list.")
    if isinstance(input_data[0], list):
        files = input_data[0]
    else:
        files = input_data
    types = []
    name, ext = os.path.splitext(files[0])
    state = int(os.path.basename(name))
    for file in files:
        name, ext = os.path.splitext(file)
        if state != int(os.path.basename(name)): break
        types.append(ext)
    return types


def list_possible_restraints():
    """Function will return a list of all possible restraint classes in Restraint.py.

    >>> biceps.toolbox.list_possible_restraints()
    """
    return [ key for key in vars(Restraint).keys() if key.startswith("Restraint_") ]

def list_possible_extensions():
    """Function will return a list of all possible input data file extensions.

    >>> biceps.toolbox.list_possible_extensions()
    """
    restraint_classes = list_possible_restraints()
    extensions = ["."+s.replace("Restraint_","") for s in restraint_classes]
    return extensions

#def list_possible_extensions():
#    """Function will return a list of all possible input data file extensions.
#
#    >>> biceps.toolbox.list_possible_extensions()
#    """
#    restraint_classes = list_possible_restraints()
#    possible = list()
#    for rest in restraint_classes:
#        print(rest)
#        R = getattr(Restraint, rest)
#        for ext in getattr(R, "_ext"):
#        #NOTE: can use _ext variable or the suffix of Restraint class
#            possible.append(ext)
#    return possible


def mkdir(path):
    """Function will create a directory if given path does not exist.

    >>> toolbox.mkdir("./doctest")
    """
    # create a directory for each system
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    """Function will remove a directory if given path exists.

    >>> toolbox.rmdir("./doctest")
    """
    # create a directory for each system
    if not os.path.exists(path):
        import shutil
        try: shutil.rmtree(path)
        except(FileNotFoundError) as e: pass


def check_indices(indices):
    if type(indices) == str:
         indices = np.loadtxt(indices)
    elif (type(indices) != np.ndarray) and (type(indices) != list):
        raise TypeError(f"Requires a path to a file (str) OR data as \
                list/np.ndarray object. You provides {type(indices)}.")
    indices = np.array(indices).astype(int)
    return indices

def check_exp_data(exp_data):
    if type(exp_data) == str:
         exp_data = np.loadtxt(exp_data)
    elif (type(exp_data) != np.ndarray) and (type(exp_data) != list):
        raise TypeError(f"Requires a path to a file (str) OR data as \
                list/np.ndarray object. You provides {type(exp_data)}.")
    return exp_data

def check_model_data(model_data):
    if type(model_data) == str:
        model_data = get_files(model_data)
    elif (type(model_data) != np.ndarray) and (type(model_data) != list):
        raise TypeError(f"Requires a path to glob files (str) OR data as \
                list/np.ndarray object. You provides {type(model_data)}.")
    return model_data




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


def plot_ref(traj, debug=True):
    """Plot reference potential for each observables.

    Args:
        traj(npz, np.array): output trajectory from BICePs sampling

    Returns:
        figure: A figure of reference potential and distribution of model observables
    """

    if debug:
        print('Loading %s ...'%traj)
    results = np.load(traj, allow_pickle=True)['arr_0'].item()
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


#TODO:
# input: pandas daraframe of populations
# output: ordered list of populations with labeled columns
def get_state_populations(file):
    """Get the state populations given a **populations.dat** file.

    Returns:
        np.ndarray: populations
    """

    return np.loadtxt(file)[:][0]

def print_top_N_pops(file, nlambda, N=5):
    pops = np.loadtxt(file)[:,nlambda-1]
    ntop = N#int(int(self.states)/10.)
    topN = pops[np.argsort(pops)[-ntop:]]
    topN_labels = [np.where(topN[i]==pops)[0][0] for i in range(len(topN))]
    print(f"Top {ntop} states: {topN_labels}")
    print(f"Top {ntop} populations: {topN}")



def print_scores(file):
    """Get the BICePs Score given a **BS.dat** file

    Returns:
        np.ndarray or float: Scores
    """

    return np.loadtxt(file)[1,0]


def npz_to_DataFrame(file, verbose=False):
    """Converts numpy Z compressed file to Pandas DataFrame (*.pkl)

    >>> biceps.toolbox.npz_to_DataFrame(file, out_filename="traj_lambda0.00.pkl")
    """

    npz = np.load(file, allow_pickle=True)["arr_0"].item()
    if verbose: print(npz.keys())

    # get trajectory information
    traj = npz["trajectory"]
    #freq_save_traj = traj[1][0] - traj[0][0]
    traj_headers = [ header.split()[0] for header in npz["trajectory_headers"] ]
    t = {"%s"%header: [] for header in traj_headers}
    for i in range(len(traj)):
        for k,header in enumerate(traj_headers):
            t[header].append(traj[i][k])
    df = pd.DataFrame(t, columns=traj_headers)
    return df




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


def compute_distances(states, indices, outdir):
    distances = []
    indices = check_indices(indices)
    for i in range(len(states)):
        d = md.compute_distances(md.load(states[i]), indices)*10. # convert nm to Å
        np.savetxt(outdir+'/%d.txt'%i,d)
    return distances

def compute_chemicalshifts(states, temp=298.0, pH=5.0, outdir="./"):

    states = get_files(states)
    for i in range(len(states)):
        print(f"Loading {states[i]} ...")
        state = md.load(states[i], top=states[0])
        shifts = md.nmr.chemical_shifts_shiftx2(state, pH, temp)
        shifts = shifts[0].unstack(-1)
        shifts.to_pickle(outdir+"cs_state%d.pkl"%i)


def compute_nonaa_scalar_coupling(states, indices, karplus_key, outdir="./", top=None):
    """Compute J couplings for small molecules.

    :param mdtraj.Trajectory traj: Trajectory or *.pdb/*.gro files
    :param int indices: indices file for atoms
    :param list karplus_key: karplus relation for each J coupling
    :param mdtraj.Topology default=None top: topology file (only required if a trajectory is loaded)"""

    indices = check_indices(indices)
    if [type(key) for key in karplus_key if type(key)==str] == [str for i in range(len(karplus_key))]:
        raise TypeError("Each karplus key must be a string. You provided: \n%s"%(
            [type(key) for key in karplus_key if type(key)==str]))
    if len(karplus_key) != len(indices):
        raise ValueError("The number of indices must equale the number of karplus_key.")
    #states = get_files(states)
    nstates = len(states)
    for state in range(nstates):
        conf = md.load(states[state], indices)
        J = np.zeros((len(conf),len(indices)))
        karplus = KarplusRelation()
        for i in range(len(J)):
            for j in range(len(indices)):
                ri, rj, rk, rl = [conf.xyz[0,x,:] for x in indices[j]]
                model_angle = dihedral_angle(ri, rj, rk, rl)
                J[i,j] = karplus.J(angle=model_angle, key=karplus_key[j])

        np.savetxt(os.path.join(outdir,'%d.txt'%state),J)
    #return J





def get_J3_HN_HA(top,traj=None, frame=None,  model="Bax2007", outname=None, verbose=False):
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
    if outname: np.save(outname, J)
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

def compute_nonaa_Jcoupling(traj, indices, karplus_key, top=None):
    """Compute J couplings for small molecules.

    :param mdtraj.Trajectory traj: Trajectory or *.pdb/*.gro files
    :param int indices: indices file for atoms
    :param list karplus_key: karplus relation for each J coupling
    :param mdtraj.Topology default=None top: topology file (only required if a trajectory is loaded)"""

    indices = check_indices(indices)

    if [type(key) for key in karplus_key if type(key)==str] == [str for i in range(len(karplus_key))]:
        raise TypeError("Each karplus key must be a string. You provided: \n%s"%(
            [type(key) for key in karplus_key if type(key)==str]))
    if len(karplus_key) != len(indices):
        raise ValueError("The number of indices must equale the number of karplus_key.")
    if traj.endswith('.gro'):
        conf = md.load(traj)
    elif traj.endswith('.pdb'):
        conf = md.load(traj)
    else:
        if top == None:
            raise TypeError("To load a trajectory file, a topology file must be provided.")
        conf = md.load(traj,top=top)
    J = np.zeros((len(conf),len(indices)))
    karplus = KarplusRelation()
    for i in range(len(J)):
        for j in range(len(indices)):
            ri, rj, rk, rl = [conf.xyz[0,x,:] for x in indices[j]]
            model_angle = dihedral_angle(ri, rj, rk, rl)
            J[i,j] = karplus.J(angle=model_angle, key=karplus_key[j])
    return J




def get_indices(traj, top, selection_expression=None, code_expression=None,
        out=None,debug=True):
    """Get atom indices from residue index list"""

    if out == None:
        out = 'indices.dat'

    # Load in the first trajectory
    print('Reading trajectory:' '%s'%(traj))
    if not (os.path.exists(traj)):
        print('File not found! Exit...')
        exit(1)
    t = md.load(traj, top=top)
    print('done.')

    # Get the topology python object and set it as a variable for the eval(selection)
    topology = t.topology
    #selection = t.topology.select_expression(selection_expression)
    # Your new selection (sel) in a list
    #sel = eval(selection)
    if selection_expression:
        sel = t.topology.select(selection_expression)
    if code_expression:
        sel = eval(t.topology.select_expression(code_expression))

    #sel = eval(t.topology.select_expression(selection_expression))
    #print(sel)
    if debug:
        print("selection_expression = %s"%selection_expression)
        print("selection = %s"%sel)
        df = t.topology.to_dataframe()[0]
        print(df.items())
        N = 0
        for i in sel:
            print("%s %s: atom  %s"%(df.resName[i],df.resSeq[i],i))
            N+=1
        print("%s atoms selected"%N)

        exit()

    # topology to dataframe
    table = t.topology.to_dataframe()

    # Get a selection for chimera
    atoms,chimera,chain = [],[],[]
    atom = 0
    # Use the dataframe table from mdtraj to match the residue labels
  #... with their corresponding chain
    for ind in sel:
        atoms.append(t.topology.atom(int(ind)))
        chain.append(table[0]['chainID'][int(ind)])
        # Is it chain A or chain B?
        if chain[atom] == 0:
            chimera.append(str(atoms[atom]).replace("-",".A@")[3:])
        elif chain[atom] == 1:
            chimera.append(str(atoms[atom]).replace("-",".B@")[3:])
        atom += 1

    # Check to make sure that this is the selection you want.
    if debug==True:
        for i in range(0,len(sel)):
            print(t.topology.atom(int(sel[i])))
        response = int(input("Is this the correct selection? (True=1/False=0)\n"))
        if response == 0:
            print('Exit...')
            exit(1)

    np.savetxt('%s'%out,np.array(sel),fmt='%i')
    np.savetxt('residues.txt',np.array(atoms),fmt='%s')
    #np.savetxt('residues_chimera.txt',np.array(chimera),fmt='%s')


def save_object(obj, filename):
    """Saves python object as pkl file"""

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Loads a python object from pkl file"""

    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_h5_object(obj, filename):
    """Saves python object as pkl file"""

    with h5py.File(filename, 'w') as f:
        f.create_dataset(filename, obj)

def load_h5(filename):
    """Saves python object as pkl file"""

    return h5py.File(filename, 'r')
def swapXY(array):
    return [(val[1],val[0]) for val in array]

def get_seperate_columns(grid_positions, transpose=0):
    """Seperate grid into left, bottom and other"""
    x = grid_positions.copy()
    x = np.array(x)
    X,Y = 0,1
    bottom_cols = list(filter(lambda i: i[X]==np.max(x), x))
    if bottom_cols == []:
        bottom_cols = list(filter(lambda i: i[X]==0, x))
    left_cols = list(filter(lambda i: i[Y]==np.min(x), x))
    both_cols = np.concatenate([left_cols,bottom_cols])

    other_cols = []
    for pos in x:
        if [val for val in both_cols if np.array_equal(val, pos)] == []:
            other_cols.append(pos)
    if transpose:
        return swapXY(left_cols), swapXY(bottom_cols), swapXY(other_cols)
    else:
        return left_cols, bottom_cols, other_cols


def change_stat_model_name(db, sm=None, new_sm=None):
    if (sm == None) or (new_sm == None):
        raise ValueError("Must provide stat_model name (sm) and what it\
                will be changed to (new_sm).")
    stat_models = db["stat_model"].to_numpy()
    indices = np.where(db["stat_model"].to_numpy() == sm)[0]
    for index in indices:
        #row = db.iloc[[index]]
        stat_models[index] = new_sm
    db["stat_model"] = stat_models
    return db


def compute_f0(u, nblocks=10):
    """Computes f_0 = -ln <exp(-u)>.

    INPUT
    u       - a np.array of sampled -ln P values

    PARAMETERS
    nblocks - number of blocks for uncertainty analysis

    RETURNS
    f_0   - free energy -ln Z_0/Z_off
    df_0  - uncertainty from from block-avearge analysis."""

    u = np.array(u)  # make sure u is an array, not a list
    u_min = np.min(u)

    nsamples = len(u)
    blocksamples = int(nsamples/nblocks)
    f0_estimates = []
    for i in range(nblocks):
        u_block = u[i*blocksamples:(i+1)*blocksamples]
        exp_avg = np.mean( np.exp(-1.0*(u_block - u_min)))
        f0_estimates.append( u_min - np.log(exp_avg) )
        print(i, f0_estimates[-1])

    return np.mean(f0_estimates), np.std(f0_estimates)



# Information Criterion functions:{{{
def aic(llf, nobs, df_modelwc):
    """
    Akaike information criterion
    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant
    Returns
    -------
    aic : float
        information criterion
    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion
    """

    return -2.0 * llf + 2.0 * df_modelwc


def bic(llf, nobs, df_modelwc):
    """
    Bayesian information criterion (BIC) or Schwarz criterion
    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant
    Returns
    -------
    bic : float
        information criterion
    References
    ----------
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """

    return -2.0 * llf + np.log(nobs) * df_modelwc


def hqic(llf, nobs, df_modelwc):
    """
    Hannan-Quinn information criterion (HQC)
    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant
    Returns
    -------
    hqic : float
        information criterion
    References
    ----------
    Wikipedia does not say much
    """
    return -2.0 * llf + 2 * np.log(np.log(nobs)) * df_modelwc


def aicc(llf, nobs, df_modelwc):
    """
    Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant
    Returns
    -------
    aicc : float
        information criterion
    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
    """
    #return -2.0 * llf + 2.0 * df_modelwc * nobs / (nobs - df_modelwc - 1.0)
    # These might be equiv. This equation is more common:
    return -2.0 * llf + 2.0*df_modelwc + 2.0* df_modelwc * (df_modelwc + 1.0) / (nobs - df_modelwc - 1.0)
# }}}


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            #return f"{bytes:.2f}{unit}{suffix}"
            return (bytes, f"{unit}{suffix}")
        bytes /= factor

def get_mem_details():
    # get the memory details
    svmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    #vmem = rxr.get_size(svmem.used)
    #swapmem = rxr.get_size(swap.used)
    return (svmem, swap)


def convert_to_Bytes(column):
    result = []
    for tuple in column:
        #print(tuple)
        try:
            if np.isnan(tuple):# == np.NAN:
                result.append(np.NAN)
                continue
        except(Exception) as e:
            pass
        val, unit = tuple
        if unit == "MB": val = np.float(float(val)*1e6)
        if unit == "GB": val = np.float(float(val)*1e9)
        result.append(val)
    return np.array(result)


def pyObjSize(input_obj, label=None, verbose=1):
    """https://towardsdatascience.com/the-strange-size-of-python-objects-in-memory-ce87bdfbb97f
    """

    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    result = get_size(memory_size)
    if verbose:
        if label:
            print(f"Memory of {label}".split("=")[0]+f": {result[0]} {result[1]}")
        else:
            print(f"Memory of object".split("=")[0]+f": {result[0]} {result[1]}")
    return result


def time_function(func, *args, **kwargs):
    """
    Computes the time taken to execute a function.

    Parameters:
        func (callable): The function to evaluate.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        result: The result of the function call.
        elapsed_time (float): Time taken in seconds.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function {func.__name__} took {elapsed_time:.6f} seconds to execute.")
    return result, elapsed_time



if __name__ == "__main__":

    import doctest
    doctest.testmod()




