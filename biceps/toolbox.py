# -*- coding: utf-8 -*-
import os, glob, re, pickle
import numpy as np
import pandas as pd
from biceps.J_coupling import *
from biceps.KarplusRelation import KarplusRelation
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import biceps.Restraint as Restraint
from scipy.optimize import curve_fit



def sort_data(dataFiles):
    """Sorting the data by extension into lists. Data can be located in various
    directories.  Provide a list of paths where the data can be found.
    Some examples of fileextensions: {.noe,.J,.cs_H,.cs_Ha}.

    :param list dataFiles: list of strings where the data can be found
    :raises ValueError: if the data directory does not exist

    >>> biceps.toolbox.sort_data()
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


    # list for every extension; 7 possible experimental observables supported
    data = [[] for x in range(len(list_possible_extensions()))]
    # Sorting the data by extension into lists. Various directories is not an issue...
    for i in range(len(dir_list)):
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        # This convert / sorted glob is a bit fishy... needs many tests
        for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
            if not any([j.endswith(ext) for ext in list_possible_extensions()]):
                raise ValueError(f"Incompatible File extension. Use:{list_possible_extensions()}")
            else:
                for k in range(len(list_possible_extensions())):
                    if j.endswith(list_possible_extensions()[k]):
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

    from natsort import natsorted
    globbed = glob.glob(path)
    return natsorted(globbed)



#def get_files(path):
#    """Uses a sorted glob to return a list of files in numerical order.
#
#    Args:
#        path(str): path to glob (able to use *)
#
#    >>> biceps.toolbox.get_files()
#    """
#
#    convert = lambda txt: int(txt) if txt.isdigit() else txt
#    return sorted(glob.glob(path), key=lambda x:[convert(s) for s in re.split("([0-9]+)",x)])



def list_res(input_data):
    """Determine the ordering of the experimental restraints that
    will be included in sampling.

    Args:
        input_data(list): see :attr:`biceps.Ensemble.initialize_restraints`

    >>> biceps.toolbox.list_res()
    """

    scheme=[]
    for i in input_data[0]:
        if not any([i.endswith(ext) for ext in list_possible_extensions()]):
            raise ValueError(f"Incompatible File extension. Use:{list_possible_extensions()}")
        else:
            scheme.append(i.split(".")[-1])
    return scheme

def list_extensions(input_data):
    """Determine the ordering of the experimental restraints that
    will be included in sampling.

    Args:
        input_data(list): see :attr:`biceps.Ensemble.initialize_restraints`

    >>> biceps.toolbox.list_extensions()

    """

    return [ res.split("_")[-1] for res in list_res(input_data) ]


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
    possible = list()
    for rest in restraint_classes:
        R = getattr(Restraint, rest)
        for ext in getattr(R, "_ext"):
        #NOTE: can use _ext variable or the suffix of Restraint class
            possible.append(ext)
    return possible


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




def mkdir(path):
    """Function will create a directory if given path does not exist.

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


def npz_to_DataFrame(file, out_filename="traj_lambda0.00.pkl", verbose=False):
    """Converts numpy Z compressed file to Pandas DataFrame (*.pkl)

    >>> biceps.toolbox.npz_to_DataFrame(file, out_filename="traj_lambda0.00.pkl")
    """

    npz = np.load(file, allow_pickle=True)["arr_0"].item()
    if verbose:
        print(npz.keys())

    # get trajectory information
    traj = npz["trajectory"]
    #freq_save_traj = traj[1][0] - traj[0][0]
    traj_headers = [ header.split()[0] for header in npz["trajectory_headers"] ]
    t = {"%s"%header: [] for header in traj_headers}
    for i in range(len(traj)):
        for k,header in enumerate(traj_headers):
            t[header].append(traj[i][k])
    df = pd.DataFrame(t, columns=traj_headers)
    df.to_pickle(out_filename)
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


def get_allowed_parameters(traj,rest_type=None):
    """Get nuisance parameters range.

    :param traj: output trajectory from BICePs sampling
    :var default=None rest_type: experimental restraint type
    :return list: A list of all nuisance parameters range
    """

    if not traj.endswith('.npz'):
        raise TypeError("trajectory file should be in the format of '*npz'")
    else:
        t = np.load(traj)['arr_0'].item()
        parameters = []
        if rest_type == None:
            rest_type = get_rest_type(traj)
        if 'gamma' in rest_type:
            for i in range(len(rest_type)):
                if i == len(rest_type)-1:   # means it is gamma
                    parameters.append(t['allowed_gamma'])
                else:
                    parameters.append(t['allowed_sigma'][i])
        else:
            parameters.append(t['allowed_sigma'])[i]
    return parameters


def get_sampled_parameters(traj,rest_type=None,allowed_parameters=None):
    """Get sampled parameters along time (steps).

    :param traj: output trajectory from BICePs sampling
    :var default=None rest_type: experimental restraint type
    :return list: A list of all nuisance paramters sampled
    """

    if not traj.endswith('.npz'):
        raise TypeError("trajectory file should be in the format of '*npz'")
    else:
        t = np.load(traj)['arr_0'].item()
        parameters = []
        if rest_type == None:
            rest_type = get_rest_type(traj)
        parameters = [[] for i in range(len(rest_type))]
        if allowed_parameters == None:
            allowed_parameters = get_allowed_parameters(traj,rest_type=rest_type)
        if 'gamma' in rest_type:
            for i in range(len(rest_type)):
                if i == len(rest_type)-1:   # means it is gamma
                    for j in range(len(t['trajectory'])):
                        parameters[i].append(allowed_parameters[i][t['trajectory'][j][4][i-1][1]])
                else:
                    for j in range(len(t['trajectory'])):
                        parameters[i].append(allowed_parameters[i][t['trajectory'][j][4][i][0]])
        else:
            for j in range(len(t['trajectory'])):
                parameters[i].append(allowed_parameters[i][t['trajectory'][4][j][i][0]])
    return parameters


def g(f, max_tau=10000, normalize=True):
    """Calculate the autocorrelaton function for a time-series f(t).
    INPUT
    f         - a 1D numpy array containing the time series f(t)

    PARAMETERS
    max_tau   - the maximum autocorrelation time to consider.
    normalize - if True, return g(tau)/g[0]

    RETURNS
    result    - a numpy array of size (max_tau+1,) containing g(tau).
    """

    f_zeroed = f-f.mean()
    T = f_zeroed.shape[0]
    result = np.zeros(max_tau+1)
    for tau in range(max_tau+1):
        result[tau] = np.dot(f_zeroed[0:-1-tau],f_zeroed[tau:-1])/(T-tau)

    if normalize:
        return result/result[0]
    else:
        return result


def single_exp_decay(x, a0, a1, tau1):
    return a0 + a1*np.exp(-(x/tau1))

def double_exp_decay(x, a0, a1, a2, tau1, tau2):
    return a0 + a1*np.exp(-(x/tau1)) + a2*np.exp(-(x/tau2))

def exponential_fit(ac, use_function='single'):
    """Perform a single- or double- exponential fit on an autocorrelation curve.

    Args:
        ac(np.ndarray): autocorrelation
        use_function(str): 'single' or 'double'

    Returns:
        yFit_data(np.ndarray): the y-values of the fit curve.
    """

    nsteps = ac.shape[0]
    if use_function == 'single':
        v0 = [0.0, 1.0 , 4000.]  # Initial guess [a0, a1, tau1] for a0 + a1*exp(-(x/tau1))
        popt, pcov = curve_fit(single_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
        yFit_data = single_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2])
        # print('best-fit a0 = ', popt[0], '+/-', pcov[0][0])
        # print('best-fit a1 = ', popt[1], '+/-', pcov[1][1])
        print('best-fit tau1 = ', popt[2], '+/-', pcov[2][2])
    else:
        v0 = [0.0, 0.9, 0.1, 4000., 200.0]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
        popt, pcov = curve_fit(double_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
        yFit_data = double_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2], popt[3], popt[4])
        # print('best-fit a0 = ', popt[0], '+/-', pcov[0][0])
        #print('best-fit a1 = ', popt[1], '+/-', pcov[1][1])
        #print('best-fit a2 = ', popt[2], '+/-', pcov[2][2])
        print('best-fit tau1 = ', popt[3], '+/-', pcov[3][3])
        print('best-fit tau2 = ', popt[4], '+/-', pcov[4][4])
    return yFit_data


def autocorr_valid(x,tau):
    """Cross-correlation of two 1-dimensional sequences.

    :var x: 1-dimensional sequence
    :var tau: lagtime
    """

    t = tau
    y = x[:np.size(x)-t]
    g = np.correlate(x, y, mode='valid')
    n = np.array([np.size(x)-t]*len(g))
    return g/n


def compute_ac(traj,tau,rest_type=None,allowed_parameters=None):
    """Compute auto-correlation time for sampled trajectory of nuisance parameters.

    :param traj: output trajectory from BICePs sampling
    :var tau: lagtime
    :var default=None rest_type: experimental restraint type
    :var default=None allowed_parameters: nuisacne parameters range
    :return list: a list of auto-correlation results for all nuisacne parameters
    :return figure: A figure of auto-correlation results for all nuisance parameters
    """

    if not traj.endswith('.npz'):
        raise TypeError("trajectory file should be in the format of '*npz'")
    else:
        if rest_type == None:
            rest_type = get_rest_type(traj)
        elif allowed_parameters == None:
            allowed_parameters = get_allowed_parameters(traj,rest_type=rest_type)
        else:
            sampled_parameters = [[] for i in range(len(rest_type))]
            t = np.load(traj)['arr_0'].item()['trajectory']
            if 'gamma' in rest_type:
                for i in range(len(t)):
                    for j in range(len(rest_type)):
                        if j == len(rest_type)-1:   # means it is gamma
                            sampled_parameters[j].append(allowed_parameters[j][t[i][4:][0][j-1][1]])
                        else:
                            sampled_parameters[j].append(allowed_parameters[j][t[i][4:][0][j][0]])
            else:
                for i in range(len(t)):
                    for j in range(len(rest_type)):
                        sampled_parameters[j].append(allowed_parameters[j][t[i][4:][0][j][0]])
            #ac_parameters=[[] for i in range(len(rest_type))]
            ac_parameters=[]
            for i in range(len(rest_type)):
                ac_parameters.append(autocorr_valid(np.array(sampled_parameters[i]),tau))
    n_rest = len(rest_type)
    time_in_steps = np.arange(1,len(ac_parameters[0])+1,1)
    colors = ['red','blue','green','black','magenta','gold','navy']
    plt.figure(figsize=(10,n_rest*5))
    for i in range(n_rest):
        plt.subplot(n_rest,1,i+1)
        plt.plot(time_in_steps,ac_parameters[i],label=rest_type[i],color=colors[i])
        plt.xlabel(r'$\tau$ (steps)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('autocorrelation.pdf')
    return ac_parameters

def plot_ac(ac_paramters,rest_type):
    """Plot auto-correlation results.

    :var ac_parameters: computed auto-correlation results
    :var rest_type: experimental restraint type
    :return figure: A figure of auto-correlation results for all nuisance parameters
    """

    n_rest = len(rest_type)
    time_in_steps = np.arange(1,n_rest+1,1)
    colors = ['red','blue','green','black','magenta','gold','navy']
    plt.figure(figsize=(10,n_rest*5))
    for i in range(n_rest):
        plt.subplot(n_rest,1,i+1)
        plt.plot(time_in_steps,ac_parameters[i],label=rest_type[i],color=colors[i])
        plt.xlabel(r'$\tau$ (steps)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('autocorrelation.pdf')


def compute_JSD(T1,T2,T_total,rest_type,allowed_parameters):
    """Compute JSD for a given part of trajectory.

    :var T1, T2, T_total: part 1, part2 and total (part1 + part2)
    :var rest_type: experimental restraint type
    :var allowed_parameters: nuisacne parameters range
    :return float: Jensen–Shannon divergence
    """

    restraints = rest_type
    all_JSD = np.zeros(len(restraints))
    if 'gamma' in rest_type:
        for i in range(len(restraints)):
            r1,r2,r_total = np.zeros(len(allowed_parameters[i])),np.zeros(len(allowed_parameters[i])),np.zeros(len(allowed_parameters[i]))
            if i == len(rest_type) - 1:    # means it is gamma
                for j in T1:
                    r1[j[4][i-1][1]]+=1
                for j in T2:
                    r2[j[4][i-1][1]]+=1
                for j in T_total:
                    r_total[j[4][i-1][1]]+=1
            else:
                for j in T1:
                    r1[j[4][i][0]]+=1
                for j in T2:
                    r2[j[4][i][0]]+=1
                for j in T_total:
                    r_total[j[4][i][0]]+=1
            N1=sum(r1)
            N2=sum(r2)
            N_total = sum(r_total)
            H1 = -1.*r1/N1*np.log(r1/N1)
            H1 = sum(np.nan_to_num(H1))
            H2 = -1.*r2/N2*np.log(r2/N2)
            H2 = sum(np.nan_to_num(H2))
            H = -1.*r_total/N_total*np.log(r_total/N_total)
            H = sum(np.nan_to_num(H))
            JSD = H-(N1/N_total)*H1-(N2/N_total)*H2
            all_JSD[i] = JSD
    else:
        for i in range(len(restraints)):
            r1,r2,r_total = np.zeros(len(allowed_parameters[i])),np.zeros(len(allowed_parameters[i])),np.zeros(len(allowed_parameters[i]))
            for j in T1:
                r1[j[4:][0][i][0]]+=1
            for j in T2:
                r2[j[4:][0][i][0]]+=1
            for j in T_total:
                r_total[j[4:][0][i][0]]+=1
            N1=sum(r1)
            N2=sum(r2)
            N_total = sum(r_total)
            H1 = -1.*r1/N1*np.log(r1/N1)
            H1 = sum(np.nan_to_num(H1))
            H2 = -1.*r2/N2*np.log(r2/N2)
            H2 = sum(np.nan_to_num(H2))
            H = -1.*r_total/N_total*np.log(r_total/N_total)
            H = sum(np.nan_to_num(H))
            JSD = H-(N1/N_total)*H1-(N2/N_total)*H2
            all_JSD[i] = JSD
    return all_JSD



def plot_conv(all_JSD,all_JSDs,rest_type):
    """Plot Jensen–Shannon divergence (JSD) distribution for convergence check.

    :var all_JSD: JSDs for different amount of total dataset
    :var all_JSDs: JSDs for different amount of total dataset from bootstrapping
    :var rest_type: experimental restraint type
    :return figure: A figure of JSD and JSDs distribution
    """

    fold = len(all_JSD)
    rounds = len(all_JSDs[0])
    n_rest = len(rest_type)
    new_JSD = [[] for i in range(n_rest)]
    for i in range(len(all_JSD)):
        for j in range(n_rest):
            new_JSD[j].append(all_JSD[i][j])
    JSD_dist = [[] for i in range(n_rest)]
    JSD_std = [[] for i in range(n_rest)]
    for rest in range(n_rest):
        for f in range(fold):
            temp_JSD = all_JSDs[f][:,rest]
            JSD_dist[rest].append(np.mean(temp_JSD))
            JSD_std[rest].append(np.std(temp_JSD))
    plt.figure(figsize=(10,5*n_rest))
    x = np.arange(100./fold,101.,fold)
    colors = ['red','blue','green','black','magenta','gold','navy']
    for i in range(n_rest):
        plt.subplot(n_rest,1,i+1)
        plt.plot(x,new_JSD[i],'o-',color=colors[i],label=rest_type[i])
        plt.hold(True)
        plt.plot(x,JSD_dist[i],'o',color=colors[i],label=rest_type[i])
        plt.fill_between(x,np.array(JSD_dist[i])+np.array(JSD_std[i]),np.array(JSD_dist[i])-np.array(JSD_std[i]),color=colors[i],alpha=0.2)
        plt.xlabel('dataset (%)')
        plt.ylabel('JSD')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('convergence.pdf')

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


def compute_distances(states, indices, outdir):
    """Function that uses MDTraj to compute distances given index pairs.
    Args:
        states(list): list of conformatonal state topology files
        indices(str): relative path and filename of indices for pair distances
        outdir(str): relative path for output directory

    Returns:
        distances(np.ndarray): computed distances
    """

    distances = []
    if (type(indices) != np.ndarray) and (type(indices) != list):
         indices = np.loadtxt(indices)
    indices = np.array(indices).astype(int)

    for i in range(len(states)):
        d = md.compute_distances(md.load(states[i]), indices)*10. # convert nm to Å
        np.savetxt(outdir+'/%d.txt'%i,d)
    return distances

def compute_chemicalshifts(states, temp=298.0, pH=5.0, outdir="./"):
    """Chemical shifts are computed using MDTraj, which uses ShiftX2.

    Args:
        states(list): list of conformatonal state topology files
        temp(float): solution temperature
        pH(float): pH of solution
        outdir(str): relative path for output directory
    """

    states = get_files(states)
    for i in range(len(states)):
        print(f"Loading {states[i]} ...")
        state = md.load(states[i], top=states[0])
        shifts = md.nmr.chemical_shifts_shiftx2(state, pH, temp)
        out = outdir+"cs_state%d.txt"%i
        np.savetxt(out, shifts.mean(axis=1))
        print(f"Saving {out} ...")
        #out = out.replace(".txt", ".pkl")
        #shifts.to_pickle(out)


def compute_nonaa_scalar_coupling(states, indices, karplus_key, outdir="./", top=None):
    """Compute J couplings for small molecules.

    :param mdtraj.Trajectory traj: Trajectory or *.pdb/*.gro files
    :param int indices: indices file for atoms
    :param list karplus_key: karplus relation for each J coupling
    :param mdtraj.Topology default=None top: topology file (only required if a trajectory is loaded)"""


    #ind = np.loadtxt(indices, dtype=int)
    if (type(indices) != np.ndarray) and (type(indices) != list):
         indices = np.loadtxt(indices)
    indices = np.array(indices).astype(int)


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

        np.savetxt(outdir+'%d.txt'%state,J)
    #return J



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

def compute_nonaa_Jcoupling(traj, indices, karplus_key, top=None):
    """Compute J couplings for small molecules.

    :param mdtraj.Trajectory traj: Trajectory or *.pdb/*.gro files
    :param int indices: indices file for atoms
    :param list karplus_key: karplus relation for each J coupling
    :param mdtraj.Topology default=None top: topology file (only required if a trajectory is loaded)"""

    if (type(indices) != np.ndarray) and (type(indices) != list):
         indices = np.loadtxt(indices)
    indices = np.array(indices).astype(int)


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
    """Saves python object as pickle file.
    Args:
        obj(object): python object
        filename(str): relative path for ouput

    >>> biceps.toolbox.save_object()
    """

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)






if __name__ == "__main__":

    import doctest
    doctest.testmod()




