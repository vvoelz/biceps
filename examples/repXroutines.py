
# Python libraries:{{{
import numpy as np
import string,re
import sys, time, os, gc, psutil
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import scipy
from scipy.stats import maxwell
from sklearn import metrics
#from scipy import stats
import biceps
from IPython.display import display, HTML
#from biceps import compile_biceps_results as cbr
pd.options.display.max_columns = 25
pd.options.display.max_rows = 25
pd.options.display.max_colwidth = 100
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
#biceps.toolbox.mkdir("figures")
from sklearn import metrics
import uncertainties as u
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm

#:}}}

# Methods:{{{
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




def normalize(v, order=1, axis=0):
    norm = np.atleast_1d(np.linalg.norm(v, ord=order, axis=axis))
    norm[norm==0] = 1
    #if norm == 0: return v
    return v / np.expand_dims(norm, axis)


# Use this:{{{
def get_boltzmann_weighted_states(nstates, σ=0.16, loc=3.15,
                scale=1.0, verbose=False, plot=True, domain=None):
    """Get Boltzmann weighted states and perturbed states with error equal to σ.

    Args:
        nstates(int): number of states
        σ(float): sigma is the error in the prior

    Returns:
        botlzmann weighted states and perturbed states inside tuple \
                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
    """
    import scipy.stats as stats
    iter = True

    kT = 1#0.5959 # kcal/mol
    pops = np.random.random(nstates)
    pops /= pops.sum()
    energies = -kT*np.log(pops)
    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
    df = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
    while iter == True:
        samples = df.copy()
        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
        #perturbed_w = w+σ*np.random.randn(len(w))
        perturbed_w = w+np.random.normal(loc=0.06, scale=σ, size=len(w))
        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
        perturbed_w /= perturbed_w.sum()
        perturbed_E = -kT*np.log(perturbed_w)
        if all(i >= 0.0 for i in perturbed_w): iter = False
        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
        except(Exception) as e: iter = True
    if verbose:
        print(f"Populations:           {w}")
        print(f"Perturbed Populations: {perturbed_w}")
        print(f"Energies:              {E}")
        print(f"Perturbed Energies:    {perturbed_E}")
        print(f"RMSE in populations:   {RMSE_pops}")
        print(f"RMSE in energies:      {RMSE_E}")
        #print(f"RMSE in prior:      {RMSE_E}")

    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]
#:}}}

## Good version:{{{
#def get_boltzmann_weighted_states(nstates, σ=0.16, domain=(0.15, 0.35), loc=3.15, scale=1.0, verbose=False, plot=True):
#    """Get Boltzmann weighted states and perturbed states with error equal to σ.
#
#    Args:
#        nstates(int): number of states
#        σ(float): sigma is the error in the prior
#
#    Returns:
#        botlzmann weighted states and perturbed states inside tuple \
#                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
#    """
#    import scipy.stats as stats
#    iter = True
#
#    kT = 1 #0.5959 # kcal/mol
#    energies = np.sort(np.random.uniform(domain[0], domain[1], nstates))
#    #energies = np.random.random(nstates)
#    pops = np.exp(-energies/kT)
#    pops /= pops.sum()
#    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
#    df = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
#
#    while iter == True:
#        samples = df.copy()
#        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
##        perturbed_E = E+σ*np.random.randn(len(E))
##        perturbed_w = np.exp(-perturbed_E/kT)
##        perturbed_w /= perturbed_w.sum()
#
#        perturbed_w = w+σ*np.random.randn(len(w))
#        perturbed_w /= perturbed_w.sum()
#        perturbed_E = -kT*np.log(perturbed_w)
#
#        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
#        if all(i >= 0.0 for i in perturbed_w): iter = False
#        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
#        except(Exception) as e: iter = True
#    if verbose:
#       print(f"Populations:           {w}")
#       print(f"Perturbed Populations: {perturbed_w}")
#       print(f"Energies:              {E}")
#       print(f"Perturbed Energies:    {perturbed_E}")
#       print(f"RMSE in populations:   {RMSE_pops}")
#       print(f"RMSE in energies:      {RMSE_E}")
#       #print(f"RMSE in prior:      {RMSE_E}")
#
#    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]
##:}}}
#
#
## Kinda Good version:{{{
#def get_boltzmann_weighted_states(nstates, σ=0.16, domain=(0.15, 0.35), loc=3.15, scale=1.0, verbose=False, plot=True):
#    """Get Boltzmann weighted states and perturbed states with error equal to σ.
#
#    Args:
#        nstates(int): number of states
#        σ(float): sigma is the error in the prior
#
#    Returns:
#        botlzmann weighted states and perturbed states inside tuple \
#                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
#    """
#    import scipy.stats as stats
#    iter = True
#
#    kT = 1 #0.5959 # kcal/mol
#    #energies = np.sort(np.random.uniform(domain[0], domain[1], nstates))
#    #energies = np.random.random(nstates)
#    pops = np.random.random(nstates)
#    pops /= pops.sum()
#    energies = -kT*np.log(pops)
#    pops = np.exp(-energies/kT)
#    pops /= pops.sum()
#    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
#    df = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
#
#    while iter == True:
#        samples = df.copy()
#        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
#        perturbed_w = w.copy()
#        #perturbed_w += σ*np.random.randn(len(w))
#        perturbed_w += np.random.normal(loc=0.06, scale=σ, size=len(w))
#        perturbed_E = -kT*np.log(perturbed_w)
#        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
#        try:
#            RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
#            iter = False
#        except(Exception) as e: iter = True
#    if verbose:
#       print(f"Populations:           {w}")
#       print(f"Perturbed Populations: {perturbed_w}")
#       print(f"Energies:              {E}")
#       print(f"Perturbed Energies:    {perturbed_E}")
#       print(f"RMSE in populations:   {RMSE_pops}")
#       print(f"RMSE in energies:      {RMSE_E}")
#       #print(f"RMSE in prior:      {RMSE_E}")
#
#    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]
##:}}}

## Tesing version:{{{
#def get_boltzmann_weighted_states(nstates, σ=0.16, domain=(0.15, 0.35), loc=3.15, scale=1.0, verbose=False, plot=True):
#    """Get Boltzmann weighted states and perturbed states with error equal to σ.
#
#    Args:
#        nstates(int): number of states
#        σ(float): sigma is the error in the prior
#
#    Returns:
#        botlzmann weighted states and perturbed states inside tuple \
#                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
#    """
#    import scipy.stats as stats
#    iter = True
#
#    kT = 1 #0.5959 # kcal/mol
#    #energies = np.sort(np.random.uniform(domain[0], domain[1], nstates))
#    #energies = np.random.random(nstates)
#    pops = np.random.random(nstates)
#    pops /= pops.sum()
#    energies = -kT*np.log(pops)
#    pops = np.exp(-energies/kT)
#    pops /= pops.sum()
#    E,w = energies,pops
#    perturbed_w = w
#    perturbed_E = E
#    RMSE_pops = 0
#    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]
##:}}}
#

##:IDK{{{
#
#def get_boltzmann_weighted_states(nstates, σ=0.16, domain=(0.1, 0.8), loc=0,
#        scale=1.0, verbose=False, plot=True):
#    """Get Boltzmann weighted states and perturbed states with error equal to σ.
#    Args:
#        nstates(int): number of states
#        σ(float): sigma is the error in the prior
#    Returns:
#        botlzmann weighted states and perturbed states inside tuple \
#                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
#    """
#    import scipy.stats as stats
#    iter = True
#
#    kT = 1#0.5959 # kcal/mol
#    frozen = maxwell(loc=loc, scale=scale)
#    energies = np.sort(np.random.uniform(domain[0], domain[1], nstates))
#    pops = frozen.pdf(energies)
#    pops /= pops.sum()
#    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
#    while iter == True:
#        samples = df
#        if plot:
#            ax = df.plot(x="E", y="Pops", kind="line", color="k", figsize=(14, 6))
#            fig = ax.get_figure()
#            # sample energies from Boltzmann distribution
#            for i in samples.to_numpy():
#                ax.axvline(x=i[0], ymin=0, ymax=1, color="red")#print(energies)
#            fig.savefig("boltzmann_energies.png")
#
#        samples = samples.sort_values(["Pops"], ascending=False).reset_index(drop=True)
#        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
#
#        perturbed_E = E+σ*np.random.randn(len(E))
#        perturbed_w = np.exp(-perturbed_E/kT)
#        perturbed_w /= perturbed_w.sum()
#
#        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
#        if all(i >= 0.0 for i in perturbed_w): iter = False
#        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
#        except(Exception) as e: iter = True
#
#    if verbose:
#        print(f"Populations:           {w}")
#        print(f"Perturbed Populations: {perturbed_w}")
#        print(f"Std Error per state (Pops): {np.std([w.transpose(), perturbed_w.transpose()], axis=0)}")
#        print(f"Avg Std Error per state (Pops):     {np.std([w.transpose(), perturbed_w.transpose()], axis=0).mean()}")
#        print(f"Avg Std Error per state (Energies): {np.std([E.transpose(), perturbed_E.transpose()], axis=0).mean()}")
#        print(f"RMSE in populations:   {RMSE_pops}")
#        print(f"RMSE in energies:      {RMSE_E}")
#
#    return [[E,w], [perturbed_E,perturbed_w], RMSE_E]
#
##:}}}
#


## backup 10/15/21:{{{
#
#def get_boltzmann_weighted_states(nstates, σ=0.16, scale=0.35,
#        domain=(0.1, 2.), loc=0.0, verbose=False, plot=True):
#    """Get Boltzmann weighted states and perturbed states with error equal to σ.
#
#    Args:
#        nstates(int): number of states
#        σ(float): sigma is the error in the prior
#
#    Returns:
#        botlzmann weighted states and perturbed states inside tuple \
#                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
#    """
#    import scipy.stats as stats
#    iter = True
#
#    kT = 1#0.5959 # kcal/mol
#    maxwell = stats.maxwell
#    N = 1000
#    mean, var, skew, kurt = maxwell.stats(moments='mvsk')
#    #energies = np.linspace(-1, 0, N)
#    #pops = normalize(maxwell.pdf(energies, loc=-1.1, scale=1))[0]
#    energies = np.linspace(domain[0], domain[1], N)
#    pops = normalize(maxwell.pdf(energies, loc=loc, scale=scale))[0]
#
#    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
#    while iter == True:
#        #samples = df.sample(n=nstates, weights="Pops").reset_index(drop=True)
#        samples = np.random.choice(df.index, size=nstates, replace=False)
#        samples = df.iloc[samples]
#
#        if plot:
#            ax = df.plot(x="E", y="Pops", kind="line", color="k", figsize=(14, 6))
#            fig = ax.get_figure()
#            # sample energies from Boltzmann distribution
#            for i in samples.to_numpy():
#                ax.axvline(x=i[0], ymin=0, ymax=1, color="red")#print(energies)
#            fig.savefig("boltzmann_energies.png")
#
#        samples = samples.sort_values(["Pops"], ascending=False).reset_index(drop=True)
#        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
#        w /= w.sum()
#
#        perturbed_E = E+σ*np.random.randn(len(w))
#        perturbed_w = np.exp(-perturbed_E/kT)
#        perturbed_w /= perturbed_w.sum()
#        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
#        if all(i >= 0.0 for i in perturbed_w): iter = False
#        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
#        except(Exception) as e: iter = True
#    print(f"Energies:           {E}")
#    print(f"Perturbed Energies: {perturbed_E}")
#    print(f"Populations:           {w}")
#    print(f"Perturbed Populations: {perturbed_w}")
#    if verbose:
#        print(f"RMSE in populations: {RMSE_pops}")
#        print(f"RMSE in energies: {RMSE_E}")
#    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]
## }}}
#
# Old functions for generating populations and energies:{{{
'''

    import scipy.stats as stats
    kT = 1.0 #0.5959 # kcal/mol
    if nstates <= 20:
        maxwell = stats.maxwell
        N = 1000
        mean, var, skew, kurt = maxwell.stats(moments='mvsk')
        energies = np.linspace(4, 6, N)
        pops = normalize(maxwell.pdf(energies, loc=3., scale=2.0*σ))[0]
        df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
        samples = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
        E,w = samples.to_numpy().transpose()
        samples = np.random.choice(df.index, size=nstates, replace=False)
        samples = df.iloc[samples]
        samples = samples.sort_values(["Pops"], ascending=False).reset_index(drop=True)
        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
    else:
        #E = np.random.uniform(low=-1.5, high=1.5, size=(nstates)) # np.random.random(nstates)

        E = np.random.uniform(4., 6.5, nstates)
        #E = np.linspace(4, 6.5, nstates)+σ*np.random.randn(nstates) # for 5 states
        #E = np.random.random(nstates)
        w = np.exp(-E/kT)
        df = pd.DataFrame([{"E":E[i], "Pops":w[i]} for i in range(len(E))])
        samples = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
        E,w = samples.to_numpy().transpose()

    w /= w.sum()

    x = 0
    iter = True
    while iter == True:
        x += 1
        if nstates <= 20: perturbed_w = w+σ*np.random.randn(len(w))
        else: perturbed_w = w+np.random.lognormal(mean=w.mean(), sigma=σ, size=len(w))
        perturbed_w /= perturbed_w.sum()
        print(perturbed_w)
        if any(i < 0.0 for i in perturbed_w): iter = True; print(x);continue
        perturbed_E = -kT*np.log(perturbed_w)
        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
        if all(i >= 0.0 for i in perturbed_w): iter = False
        try: RMSE_E = np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))
        except(Exception) as e: iter = True

    print(f"Energies:           {E}")
    print(f"Perturbed Energies: {perturbed_E}")
    print(f"Populations:           {w}")
    print(f"Perturbed Populations: {perturbed_w}")
    if verbose:
        #print(f"Std Error per state (Pops): {np.std([w.transpose(), perturbed_w.transpose()], axis=0)}")
        #print(f"Avg Std Error per state (Pops):     {np.std([w.transpose(), perturbed_w.transpose()], axis=0).mean()}")
        #print(f"Avg Std Error per state (Energies): {np.std([E.transpose(), perturbed_E.transpose()], axis=0).mean()}")
        print(f"RMSE in populations: {RMSE_pops}")
        print(f"RMSE in energies: {RMSE_E}")
        #print(f"RMSE in energies: {np.sqrt(metrics.mean_squared_error(E.transpose(), perturbed_E.transpose()))}")
    #exit()
    return [[E,w], [perturbed_E,perturbed_w], RMSE_pops]

'''


#def get_boltzmann_weighted_states(nstates, σ=0.16, verbose=False):
#    """Get Boltzmann weighted states and perturbed states with error equal to σ.
#
#    Args:
#        nstates(int): number of states
#        σ(float): sigma is the error in the prior
#
#    Returns:
#        botlzmann weighted states and perturbed states inside tuple \
#                ((energies(np.ndarray),pops(np.ndarray)),(perturbed_energies(np.ndarray),perturbed_pops(np.ndarray))
#    """
## OLD:{{{
##    kT = 0.5959 # kcal/mol
##    E = np.random.random(nstates)
##    w = np.exp(-E/kT)
##    w /= w.sum()
##    df = pd.DataFrame([{"E":E[i], "Pops":w[i]} for i in range(len(E))])
##    samples = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
##    E,w = samples.to_numpy().transpose()
##    perturbed_E = E+σ*np.random.randn(len(E))
##    perturbed_w = np.exp(-perturbed_E/kT)#/sum(np.exp(-perturbed_E*kbT))
##    perturbed_w /= perturbed_w.sum()
#
#
#
##    notNAN = False
##    while notNAN == False:
##        kT = 1#0.5959 # kcal/mol
##        maxwell = stats.maxwell
##        N = 5000
##        kT = 1
##        mean, var, skew, kurt = maxwell.stats(moments='mvsk')
##        energies = np.linspace(0, 1, N)
##        pops = maxwell.pdf(energies, loc=0., scale=0.15)
##        df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
##        samples = df.sample(n=nstates, weights="Pops").reset_index(drop=True)
##        samples = samples.sort_values(["Pops"], ascending=False).reset_index(drop=True)
##        E,w = samples.to_numpy().transpose()
##        w /= w.sum()
##        perturbed_w = w+σ*np.random.randn(len(w))
##        perturbed_w /= perturbed_w.sum()
##        perturbed_E = -kT*np.log(perturbed_w)
##        for energy in (perturbed_E-perturbed_E.min()):
##            if np.isnan(energy): continue
##            else: notNAN = True
#
##:}}}
#
#    import scipy.stats as stats
#    iter = True
#
##    kT = 1#0.5959 # kcal/mol
##    maxwell = stats.maxwell
##    N = 100000
##    mean, var, skew, kurt = maxwell.stats(moments='mvsk')
##    energies = np.linspace(-1, 0, N)
##    pops = normalize(maxwell.pdf(energies, loc=-1.1, scale=1))[0]
##    df = pd.DataFrame([{"E":energies[i], "Pops":pops[i]} for i in range(len(pops))])
##    while iter == True:
##        #samples = df.sample(n=nstates, weights="Pops").reset_index(drop=True)
##        samples = np.random.choice(df.index, size=nstates, replace=False)
##        samples = df.iloc[samples]
##        plot=True
##        if plot:
##            ax = df.plot(x="E", y="Pops", kind="line", color="k", figsize=(14, 6))
##            fig = ax.get_figure()
##            # sample energies from Boltzmann distribution
##            for i in samples.to_numpy():
##                ax.axvline(x=i[0], ymin=0, ymax=1, color="red")#print(energies)
##            fig.savefig("boltzmann_energies.png")
##
##        samples = samples.sort_values(["Pops"], ascending=False).reset_index(drop=True)
##        E,w = samples["E"].to_numpy(), samples["Pops"].to_numpy()
##        w /= w.sum()
##
##        perturbed_E = E+σ*np.random.randn(len(w))
##        perturbed_w = np.exp(-perturbed_E/kT)
##        perturbed_w /= perturbed_w.sum()
##       # perturbed_w = w+σ*np.random.randn(len(w))
##       # perturbed_w /= perturbed_w.sum()
##       # perturbed_E = -kT*np.log(perturbed_w)
##
##        if all(i >= 0.0 for i in perturbed_w): iter = False
##        if perturbed_w.sum() != 1.0: iter = True
##        if np.std([E.transpose(), perturbed_E.transpose()], axis=0).mean() > σ: iter = True
#
#
#    iter = True
#    while iter == True:
#        kT = 1.#0.5959 # kcal/mol
#        E = np.random.uniform(low=-2.0, high=2.0, size=(nstates))+σ*np.random.randn(nstates)
#        #E = np.random.uniform(low=-1.0, high=1.0, size=(nstates))+σ*np.random.randn(nstates)
#        #E = np.random.uniform(low=-4.0, high=4.0, size=(nstates))+σ*np.random.randn(nstates)
#  #      E = np.random.uniform(low=-8.0, high=0.0, size=(nstates))
#        #E = np.random.uniform(low=-10.0, high=0.0, size=(nstates))
#        #E = np.random.uniform(low=-50.0, high=0.0, size=(nstates))
#        w = np.exp(-E/kT)
#        w /= w.sum()
#        E = -kT*np.log(w)
#        df = pd.DataFrame([{"E":E[i], "Pops":w[i]} for i in range(len(E))])
#        samples = df.sort_values(["Pops"], ascending=False).reset_index(drop=True)
#        E,w = samples.to_numpy().transpose()
#        #perturbed_E = E+σ*np.random.randn(len(E))
#        #perturbed_w = np.exp(-perturbed_E/kT)
#        if σ == 0.16: A = 1.5
#        elif σ == 0.08: A = 1.2
#        else: A = 1.0
#        perturbed_w = w+A*σ*np.random.randn(len(w))
#        perturbed_w /= perturbed_w.sum()
#        perturbed_E = -kT*np.log(perturbed_w)
#        RMSE_pops = np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))
#        avg_error_E = np.std([E.transpose(), perturbed_E.transpose()], axis=0).mean()
#        if all(i >= 0.0 for i in perturbed_w): iter = False
#        #if (RMSE_pops > 2.0*σ): iter = True
#        #if (avg_error_E > 2.0*σ): iter = True
#
#    print(f"Populations:           {w}")
#    print(f"Perturbed Populations: {perturbed_w}")
#    if verbose:
#        print(f"Std Error per state (Pops): {np.std([w.transpose(), perturbed_w.transpose()], axis=0)}")
#        print(f"Avg Std Error per state (Pops):     {np.std([w.transpose(), perturbed_w.transpose()], axis=0).mean()}")
#        print(f"Avg Std Error per state (Energies): {np.std([E.transpose(), perturbed_E.transpose()], axis=0).mean()}")
#        print(f"RMSE in populations: {np.sqrt(metrics.mean_squared_error(w.transpose(), perturbed_w.transpose()))}")
#    return [[E,w], [perturbed_E,perturbed_w]]
# }}}

def get_data(x, weights, Nd, μ_data, σ_data, verbose=True):

    exp = np.array([w*x[i] for i,w in enumerate(weights)]).sum(axis=0)
    _exp = exp.copy()
    if (μ_data or σ_data) != 0.0:
        if verbose: print(f"\n(μ_data, σ_data) = ({μ_data}, {σ_data})\n")
        if σ_data > 0.0:
            exp += σ_data*np.random.randn(len(exp))
        if μ_data > 0.0:
            offset = np.random.uniform(3.0, 5.0, len(exp))
            for i in range(len(exp)):
                if np.random.random() <= 0.30:
                    exp[i] = exp[i] + offset[i]

            ## NOTE: Only for 30% of the datapoints
            #samples = np.random.choice(exp, size=int(len(exp)*.30), replace=False)
            #if samples.size == 0:
            #    if np.random.random() <= 0.30:
            #        exp[0] = exp[0] + offset[0]
            #else:
            #    for i in [np.where(exp == sample) for sample in samples]:
            #        exp[i] = exp[i] + offset[i] #3.0+2.0*np.random.random()
    diff = abs(_exp - exp)
    return x,exp,diff



def gen_synthetic_data(dir, nStates, Nd, σ_prior=0, μ_prior=0, σ_data=0, μ_data=0,
        boltzmann_domain=(0.15, 0.35), boltzmann_loc=0, boltzmann_scale=1.0,
        verbose=False, plot=True, as_intensities=False):
    """Generate synthetic input files for biceps including the
    experimental NOE data weighted by all the states. To mimic random errors,
    Gaussian noise can be added by tuning the parameter σ. To mimic systematic
    errors, a shift in the data can be achieved by changing μ.
    """

    biceps.toolbox.mkdir(dir)
    #states,perturbed_states,RMSE = get_boltzmann_weighted_states(nStates, σ=σ_prior, verbose=verbose, plot=plot)
    states,perturbed_states,RMSE = get_boltzmann_weighted_states(nStates, σ=σ_prior, domain=boltzmann_domain, loc=boltzmann_loc, scale=boltzmann_scale, verbose=verbose, plot=plot)
    np.savetxt(f"{dir}/avg_prior_error.txt", np.array([RMSE]))
    energies,pops = states
    np.savetxt(f"{dir}/energies.txt",energies)
    np.savetxt(f"{dir}/pops.txt",pops)

    if σ_prior == 0.0:
        perturbed_energies,perturbed_pops = states
        RMSE = 0.0
    else:
        perturbed_energies,perturbed_pops = perturbed_states
    np.savetxt(f"{dir}/prior.txt", perturbed_energies)
    np.savetxt(f"{dir}/prior_pops.txt", perturbed_pops)

    weights = pops
    ############################################################################
    x = np.random.uniform(low=1.0, high=10.0, size=(len(weights),Nd))
    if as_intensities:
        #x = np.array([x[i]**-6 for i,w in enumerate(weights)])
        x = x**-6

    _x, exp, diff = get_data(x, weights, Nd, μ_data, σ_data, verbose=verbose)
    if verbose: print(f"Experimental errors: {diff}")
    if verbose: print(f"N data points perturbed: {np.count_nonzero(diff)}")
    # compute approximate sigma
    sse = 0
    for dev in diff: sse += dev*dev
    sigma = np.sqrt(sse/len(diff)-1)#/np.sqrt(len(diff))
    if verbose: print(f"sigma: {sigma}")
    np.savetxt(f"{dir}/data_sigma.txt", np.array([sigma]))
    np.savetxt(f"{dir}/data_deviations.txt", diff)

    dir = dir+"/NOE"
    biceps.toolbox.mkdir(dir)
    for i in range(len(weights)):
        model = pd.read_pickle("template.noe")
        _model = pd.DataFrame()
        for j in range(Nd):
            model["restraint_index"], model["model"], model["exp"] = j+1, x[i,j], exp[j]
            #_model = _model.append(model, ignore_index=True)
            _model = pd.concat([_model,model], ignore_index=True)
        _model.to_pickle(dir+"/%s.noe"%i)




def get_synthetic_data_points(dir, nStates, list_of_Nds=[20],
        σ_prior=0, σ_data=0, μ_data=0, verbose=False):
    """Generate synthetic data and input data files for both experiment and
    forward model. Use σ_prior to generate error in the prior populations.
    To mimic random errors, Gaussian noise can be added by
    tuning the parameter σ_data. To mimic systematic
    errors, a shift in the data can be achieved by changing μ_data.
    """

    boltzmann_domain,boltzmann_loc,boltzmann_scale=(0.15, 0.35),0,1.0
    states,perturbed_states,RMSE = get_boltzmann_weighted_states(nStates,
            σ=σ_prior, domain=boltzmann_domain, loc=boltzmann_loc,
            scale=boltzmann_scale, verbose=verbose, plot=False)
    energies,pops = states
    perturbed_energies,perturbed_pops = perturbed_states
    data = []
    for Nd in list_of_Nds:
        x = np.random.uniform(low=1.0, high=10.0, size=(len(pops),Nd))
        _x, exp, diff = get_data(x, pops, Nd, μ_data, σ_data, verbose=verbose)
        if verbose: print(f"Experimental errors: {diff}")
        if verbose: print(f"N data points perturbed: {np.count_nonzero(diff)}")
        # compute approximate sigma
        sse = 0
        for dev in diff: sse += dev*dev
        data_sigma = np.sqrt(sse/len(diff))
        if verbose: print(f"data_sigma: {data_sigma}")
        new_dir = dir+f"/NOE_{Nd}"
        biceps.toolbox.mkdir(new_dir)
        for i in range(len(pops)):
            model = pd.read_pickle("template.noe")
            _model = pd.DataFrame()
            for j in range(Nd):
                model["restraint_index"], model["model"], model["exp"] = j+1, x[i,j], exp[j]
                _model = _model.append(model, ignore_index=True)
            _model.to_pickle(new_dir+"/%s.noe"%i)
        data.append({"Nd": Nd,
                     "dir": new_dir,
                     "energies": energies,
                     "perturbed_energies": perturbed_energies,
                     "pops": pops,
                     "perturbed_pops": perturbed_pops,
                     "RMSE": RMSE,
                     "data_sigma": data_sigma,
                     "data_deviations": diff
                     })
    return data



def plot_distribution_of_prior_error(nStates, σ_prior, scale=1,
        domain=(0.12, 0.6), loc=0., nSamples=1000, figsize=(6, 4)):

    errors,E_errors = [],[]
    for i in tqdm(range(nSamples)):
        bws = get_boltzmann_weighted_states(nStates, σ=σ_prior,
                  scale=scale, domain=domain, loc=loc, plot=False)
        avg_prior_error = bws[-1]
        #np.sqrt(metrics.mean_squared_error(bws[0][1].T,bws[1][1].T))
        #avg_E_error = np.sqrt(metrics.mean_squared_error(bws[0][0].T,bws[1][0].T))
        errors.append({"error": avg_prior_error})
        #E_errors.append({"error": avg_E_error})
        populations = bws[0][1]
    df = pd.DataFrame(errors)
    #E_df = pd.DataFrame(E_errors)
    print("Populations RMSE = %0.3f ± %0.3f"%(df['error'].mean(), df['error'].std()))
    #print("Energies RMSE    = %0.3f ± %0.3f"%(E_df['error'].mean(), E_df['error'].std()))
    #%matplotlib inline
    ax = plt.subplot()
    fig = ax.get_figure()
    df["error"].hist(alpha=0.5, bins="auto", edgecolor='black', linewidth=1.2, color="b", label="Populations", ax=ax)
    ax.axvline(x=df['error'].mean()+ df['error'].std(), color='k', linewidth=4, linestyle="--")
    ax.axvline(x=df['error'].mean(), color='k', linewidth=4, linestyle="-")
    ax.axvline(x=df['error'].mean()+ -df['error'].std(), color='k', linewidth=4, linestyle="--")
    #E_df["error"].hist(alpha=0.5, bins="auto", edgecolor='black', linewidth=1.2, color="r", label="Energies", ax=ax)
    ax.set_xlabel(r"$\sigma_{prior}$", size=16)
    ax.set_xlim(0, 0.35)
    ax.legend()
    fig.set_size_inches(*figsize)
    fig.tight_layout()


def plot_distribution_of_data_error(data, figsize=(10, 6)):


    gs = gridspec.GridSpec(2, 2)

    #data["systematic"] = [np.mean([data["Experiment (no error)"].to_numpy()[i], data["Experiment (systematic error)"].to_numpy()[i]]) for i in range(len(data["Experiment (no error)"]))]
    data["systematic"] = [-np.subtract(data["Experiment (no error)"].to_numpy()[i], data["Experiment (systematic error)"].to_numpy()[i]) for i in range(len(data["Experiment (no error)"]))]
    data["random"] = [-np.subtract(data["Experiment (no error)"].to_numpy()[i], data["Experiment (random error)"].to_numpy()[i]) for i in range(len(data["Experiment (no error)"]))]


    ax1 = plt.subplot(gs[0,0])
    data["Experiment (systematic error)"].plot.hist(alpha=0.5, ax=ax1, edgecolor='black', linewidth=1.2, color="r", label="Systematic error")
    data["Experiment (no error)"].plot.hist(alpha=0.5, ax=ax1, edgecolor='black', linewidth=1.2, color="b", label="No error")
    ax1.set_title("Adding Systematic Error", fontsize=16)
    ax1.legend()
    ax1.set_ylabel("")
    ax1.set_xlabel(r"NOE distance ($\AA$)", size=14)
    ax1.grid()

    ax2 = plt.subplot(gs[0,1])
    data["Experiment (random error)"].plot.hist(alpha=0.5, ax=ax2, edgecolor='black', linewidth=1.2, color="r", label="Random error")
    data["Experiment (no error)"].plot.hist(alpha=0.5, ax=ax2, edgecolor='black', linewidth=1.2, color="b", label="No error")
    ax2.set_title("Adding Random Error", fontsize=16)
    ax2.legend()
    ax2.set_ylabel("")
    ax2.set_xlabel(r"NOE distance ($\AA$)", size=14)
    ax2.grid()

    ax3 = plt.subplot(gs[1,0])
    data["systematic"].plot.hist(alpha=0.5, ax=ax3, edgecolor='black', linewidth=1.2, color="b")
    ax3.set_title(r"3 to 5 $\AA$ shift to 30% of data points", fontsize=16)
    ax3.set_ylabel("")
    ax3.set_xlabel(r"NOE distance ($\AA$)", size=14)
    ax3.grid()

    ax4 = plt.subplot(gs[1,1])
    data["random"].plot.hist(alpha=0.5, ax=ax4, edgecolor='black', linewidth=1.2, color="b")
    ax4.axvline(x=data["random"].mean()+ data["random"].std(), color='k', linewidth=4, linestyle="--")
    ax4.axvline(x=data["random"].mean(), color='k', linewidth=4, linestyle="-")
    ax4.axvline(x=data["random"].mean()+ -data["random"].std(), color='k', linewidth=4, linestyle="--")
    ax4.set_title(r"$\sigma$ = %0.3f"%data["random"].std(), fontsize=16)
    ax4.set_ylabel("")
    ax4.set_xlabel(r"NOE distance ($\AA$)", size=14)
    ax4.grid()

    fig = ax1.get_figure()
    fig.set_size_inches(*figsize)
    fig.tight_layout()


#:}}}

# Append to Database:{{{
def append_to_database(A, dbName="database_Nd.pkl", verbose=False, **kwargs):
    n_lambdas = A.K
    pops = A.P_dP[:,n_lambdas-1]
    BS = A.f_df
    data = pd.DataFrame()
    data["nsteps"] = [kwargs.get("nsteps")]
    data["nstates"] = [kwargs.get("nStates")]
    data["nlambda"] = [kwargs.get("n_lambdas")]
    data["nreplica"] = [kwargs.get("nreplicas")]
    data["lambda_swap_every"] = [kwargs.get("lambda_swap_every")]
    data["Nd"] = [kwargs.get("Nd")]
    data["nparameters"] = [A.get_number_of_parameters()]
    data["prior error"] = [kwargs.get("σ_prior")]
    data["σ_data"] = [kwargs.get("σ_data")]
    data["μ_data"] = [kwargs.get("μ_data")]
    if (kwargs.get("σ_data")==0.0) and (kwargs.get("μ_data")==0.0): data["data error type"] = ["None"]
    if (kwargs.get("σ_data")>0.0) and (kwargs.get("μ_data")==0.0): data["data error type"] = ["Random"]
    if (kwargs.get("σ_data")==0.0) and (kwargs.get("μ_data")>0.0): data["data error type"] = ["Systematic"]
    if (kwargs.get("σ_data")>0.0) and (kwargs.get("μ_data")>0.0): data["data error type"] = ["Random & Systematic"]

    if "populations" in kwargs.keys():
        populations = kwargs.get("populations")
        data["prior pops"] = [populations]
    else:
        data["prior pops"] = [np.loadtxt(f"{kwargs.get('dir')}/pops.txt")]
        populations = data["prior pops"][0]

    RMSE = np.sqrt(metrics.mean_squared_error(pops, populations))
    if verbose: print(f"\n\nRMSE = {RMSE}")


    data["RMSE"] = [RMSE]
    data_uncertainty = kwargs.get("data_uncertainty")
    data["uncertainties"] = [data_uncertainty]
    data["stat_model"] = [kwargs.get("stat_model")]
    if "alpha" in kwargs.keys(): data["alpha"] = [kwargs.get("alpha")]


#    for i,lam in enumerate(kwargs.get("lambda_values")):
#        lam = "%0.2g"%lam
#        model = A.get_model_scores(model=i, verbose=False)# True)
#        data["BIC%s"%lam] = [np.mean(model["BIC"])] # NOTE: taking average (?)
#        data["llf%s"%lam] = [np.mean(model["llf"])] # NOTE: taking average (?)
#
#    for i,lam in enumerate(kwargs.get("lambda_values")):
#        lam = "%0.2g"%lam
#        data["BIC Score lam=%s"%lam] = [np.float(-0.5*(data["BIC%s"%lam]-data[f"BIC{kwargs.get('lambda_values')[0]:.2g}"]))]

    model_scores = A.get_model_scores(verbose=False)# True)
    for i,lam in enumerate(kwargs.get("lambda_values")):
        lam = "%0.2g"%lam
        data["BIC Score lam=%s"%lam] = [np.float(model_scores[i]["BIC score"])]



    for i,lam in enumerate(kwargs.get("lambda_values")):
        lam = "%0.2g"%lam
        data["BICePs Score lam=%s"%lam] = [BS[i,0]]
        data["BICePs Score Std lam=%s "%lam] = [2*BS[i,1]] # at 95% C

#    for i,lam in enumerate(kwargs.get("lambda_values")):
#        model = A.get_model_scores(model=i)
#        data["BIC%0.2g"%lam] = [model["BIC"]]
#    #data["BIC score"] = [-0.5*(BIC1-BIC0)]
#
#    for i,lam in enumerate(kwargs.get("lambda_values")):
#        lam = "%0.2g"%lam
#        data["BICePs Score lam=%s"%lam] = [BS[i,0]]
#        data["BICePs Score Std lam=%s "%lam] = [2*BS[i,1]] # at 95% C

    data["time sampled (s)"] = [kwargs.get("time sampled (s)")]
    data["vmem"] = [kwargs.get("vmem")]
    data["swapmem"] = [kwargs.get("swapmem")]
    data["pops"] = [pops]
    data["D_KL"] = [np.nansum([pops[i]*np.log(pops[i]/populations[i]) for i in range(len(pops))])]
    #data["RMSE"] = [np.sqrt(metrics.mean_squared_error(pops, populations))]
    data["k"] = [A.get_restraint_intensity()]
    try:
        data["priors"] = [np.loadtxt(f"{kwargs.get('dir')}/prior.txt")]
    except(Exception) as e:
        data["priors"] = [kwargs.get('prior')]


    if "prior_RMSE" in kwargs.keys():
        data["avg prior error"] = [kwargs.get("prior_RMSE")]
    else:
        data["avg prior error"] = [np.loadtxt(f"{kwargs.get('dir')}/avg_prior_error.txt")]
    if "change_Nr_every" in kwargs.keys(): data["change_Nr_every"] = [kwargs.get("change_Nr_every")]

    #if (("data_sigma" in kwargs.keys()) and ("data_deviations"  in kwargs.keys())):
    if "data_sigma" in kwargs.keys():
        data_sigma = kwargs.get("data_sigma")[0]
        if np.isnan(data_sigma): data_sigma = 0.0
        data_deviations = kwargs.get("data_deviations")
        mlp = pd.concat([A.get_max_likelihood_parameters(model=i) for i in range(kwargs.get("n_lambdas"))])
        mlp.reset_index(inplace=True, drop=True)
        mlp = mlp.iloc[[1]]
        columns = [col for col in mlp.columns.to_list() if "sigma" in col]
        mlp = mlp[columns]
        mlp.columns = [col.split("sigma_")[-1] for col in columns]

        sigmas = mlp.to_numpy()[0]
        if data_uncertainty == "single":
            total_sigma = sigmas[0]
            sigma_RMSE = np.abs(data_sigma - total_sigma)
        else:
            total_sigma = np.sqrt(np.sum([sigma*sigma for sigma in sigmas])/(len(sigmas)-1))
            sigma_RMSE = np.sqrt(metrics.mean_squared_error(data_deviations, sigmas))

        data["data_sigma"] = [data_sigma]
        data["sigma_RMSE"] = [sigma_RMSE]


    # NOTE: Saving results to database
    if os.path.isfile(dbName):
       db = pd.read_pickle(dbName)
    else:
        if verbose: print("Database not found...\nCreating database...")
        db = pd.DataFrame()
        db.to_pickle(dbName)
    # append to the database
    db = db.append(data, ignore_index=True)
    db.to_pickle(dbName)
    gc.collect()

# }}}

# Plotting Methods:{{{

# plot_stat_models_RMSE:{{{
def plot_stat_models_RMSE(df, stat_models, xcol="Nd", ycol="RMSE",
            xlabel=r"$N_{d}$", ylabel="RMSE", ylim=(0.0, 0.3), show_prior_error=False,
            avg_prior_error_across_all_Nds=True, figname="models_RMSE.pdf"):

    results = df.copy()
    fig = plt.figure(figsize=(8, 4*len(stat_models)))
    gs = gridspec.GridSpec(len(stat_models), 1)
    positions = [(i,0) for i in range(len(stat_models))]
    axes = []
    for i,stat_model in enumerate(stat_models):
        ax = plt.subplot(gs[positions[i]])
        results_df = results.where(results["stat_model"]==stat_model)
        nreplica = results_df["nreplica"].to_numpy(dtype=int)
        nreplica = np.bincount(nreplica).argmax()
        if nreplica == 0:
            nreplica = results_df["nreplica"].to_numpy(dtype=int).max()

        result = results_df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
        error = results_df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
        result = result.reset_index()
        error = error.reset_index()
        result01 = [x for _, x in result.groupby(['data error type', 'prior error'])]
        error01 = [x for _, x in error.groupby(['data error type', 'prior error'])]
        if show_prior_error:
            if avg_prior_error_across_all_Nds:
                ape = results_df["avg prior error"].mean()
                ape_std = results_df["avg prior error"].std()
            else:
                single_data = results_df.where(results["Nd"]==results["Nd"].to_numpy().min())
                ape = single_data["avg prior error"].mean()
                ape_std = single_data["avg prior error"].std()


        colors = {"None": "grey", "Random": "orange", 'Systematic':"g", 'Random & Systematic':"r"}
        for d,data in enumerate(result01):
            color = colors[data["data error type"].to_numpy()[0]]
            if show_prior_error:
                ax.plot(np.concatenate([np.array([0]),data[xcol].to_numpy()]),
                         np.concatenate([np.array([ape]),data[ycol].to_numpy()]),
                         label=data["data error type"].to_numpy()[0], linestyle="solid",
                         c=color)
            else:
                ax.plot(np.concatenate([data[xcol].to_numpy()]),
                         np.concatenate([data[ycol].to_numpy()]),
                         label=data["data error type"].to_numpy()[0], linestyle="solid",
                         c=color)

            ax.errorbar(np.concatenate([data[xcol].to_numpy()]),
                         np.concatenate([data[ycol].to_numpy()]),
                         yerr=np.concatenate([error01[d][ycol].to_numpy()]),
                         color=color, markersize=3,
                         ls="None", fmt='-o', capsize=3)

        if show_prior_error:
            ax.plot(np.array([0]), ape, c="k")
            ax.errorbar(np.array([0]), ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)

        ax.tick_params(which="major", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="major", axis="x", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.tick_params(which="minor", axis="y", direction="inout", left=1, bottom=1, right=1, top=1)
        ax.grid(alpha=0.5, linewidth=0.5)

        xticklabels = result01[0][xcol].to_numpy()
        ax.set_xticks(xticklabels)

        if i == (len(stat_models)-1):
            ax.set_xticklabels([f"{x}" for x in xticklabels])
            ax.set_xlabel(xlabel, fontsize=16)
        else:
            ax.xaxis.set_ticklabels([])

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(*ylim)
        ax.xaxis.label.set_size(16)
        ax.set_title(f"{stat_model} model", fontsize=16)
        #ax.legend()

        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=False, left=True, right=False)


        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(16)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=16)
                if k == 0:
                    mark.set_rotation(s=25)
        x,y = -0.1, 1.02
        ax.text(x,y, string.ascii_lowercase[i], transform=ax.transAxes,
                size=20, weight='bold')

    fig.tight_layout()
    fig.savefig(f"{figname}", dpi=600)
# }}}

# plot_all_models:{{{
def plot_all_models(df, figname, nstates, nreplicas=128, prior_error=0.08):

    # PLOT
    fig = plt.figure(figsize=(8, 16))
    gs = gridspec.GridSpec(5, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    model = "Bayesian"
    ax = plt.subplot(gs[0,0])
    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == 1)
        & ((df["stat_model"] == model))
        )]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    linestyles = ["solid", "dashdot"]
    for s,swap_every in enumerate([0, 500]):
        _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
            & (df["nreplica"] == 1) & (df["lambda_swap_every"] == swap_every)
            & ((df["stat_model"] == model))
            )]
        _df = _df.iloc[np.where(_df["uncertainties"] == "single")]
        result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
        error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
        result = result.reset_index()
        error = error.reset_index()
        result = [x for _, x in result.groupby(['data error type', 'prior error'])]
        result[-2], result[-1] = result[-1], result[-2]
        result01 = [_df for _df in result if _df["prior error"].to_numpy()[0] == prior_error]
        error = [x for _, x in error.groupby(['data error type', 'prior error'])]
        error[-2], error[-1] = error[-1], error[-2]
        error01 = [_df for _df in error if _df["prior error"].to_numpy()[0] == prior_error]

        colors = ["grey", "orange", "g", "r"]

        for i,data in enumerate(result01):
            ax.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                     label=data["data error type"].to_numpy()[0], linestyle=linestyles[s],
                     c=colors[i])
            ax.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         yerr=np.concatenate([np.array([0.02]),error01[i]["RMSE"].to_numpy()]),
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)
    ax.plot(np.array([0]), ape, c="k")
    ax.errorbar(np.array([0]), ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result01[0]["Nd"].to_numpy()
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    #ax.set_xlabel(r"$N_{d}$", fontsize=16)
    ax.xaxis.set_ticklabels([])
    ax.set_ylim(0,0.2)
    ax.xaxis.label.set_size(16)
    ax.set_title(f"{model} model", fontsize=16)
    #ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=False, left=True, right=False)


    model = "Gaussian"
    ax1 = plt.subplot(gs[1,0])
    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == nreplicas)
        & ((df["stat_model"] == model))
        )]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    linestyles = ["solid", "dashdot"]
    for s,swap_every in enumerate([0, 500]):
        _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
            & (df["nreplica"] == nreplicas) & (df["lambda_swap_every"] == swap_every)
            & ((df["stat_model"] == model))
            )]
        _df = _df.iloc[np.where(_df["uncertainties"] == "multiple")]
        result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
        error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
        result = result.reset_index()
        error = error.reset_index()
        result = [x for _, x in result.groupby(['data error type', 'prior error'])]
        result[-2], result[-1] = result[-1], result[-2]
        result01 = [_df for _df in result if _df["prior error"].to_numpy()[0] == prior_error]
        error = [x for _, x in error.groupby(['data error type', 'prior error'])]
        error[-2], error[-1] = error[-1], error[-2]
        error01 = [_df for _df in error if _df["prior error"].to_numpy()[0] == prior_error]

        colors = ["grey", "orange", "g", "r"]

        for i,data in enumerate(result01):
            ax1.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                     label=data["data error type"].to_numpy()[0], linestyle=linestyles[s],
                     c=colors[i])
            ax1.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         yerr=np.concatenate([np.array([0.02]),error01[i]["RMSE"].to_numpy()]),
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)
    ax1.plot(np.array([0]), ape, c="k")
    ax1.errorbar(np.array([0]), ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result01[0]["Nd"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylabel("RMSE", fontsize=16)
    #ax1.set_xlabel(r"$N_{d}$", fontsize=16)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylim(0,0.2)
    ax1.xaxis.label.set_size(16)
    ax1.set_title(f"{model} model", fontsize=16)
    #ax1.legend()


    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=False, left=True, right=False)

    model = "OutliersSP"
    ax2 = plt.subplot(gs[2,0])
    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == nreplicas)
        & ((df["stat_model"] == model))
        )]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    linestyles = ["solid", "dashdot"]
    for s,swap_every in enumerate([0, 500]):
        _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
            & (df["nreplica"] == nreplicas) & (df["lambda_swap_every"] == swap_every)
            & ((df["stat_model"] == model))
            )]
        _df = _df.iloc[np.where(_df["uncertainties"] == "single")]
        result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
        error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
        result = result.reset_index()
        error = error.reset_index()
        result = [x for _, x in result.groupby(['data error type', 'prior error'])]
        result[-2], result[-1] = result[-1], result[-2]
        result01 = [_df for _df in result if _df["prior error"].to_numpy()[0] == prior_error]
        error = [x for _, x in error.groupby(['data error type', 'prior error'])]
        error[-2], error[-1] = error[-1], error[-2]
        error01 = [_df for _df in error if _df["prior error"].to_numpy()[0] == prior_error]

        colors = ["grey", "orange", "g", "r"]

        for i,data in enumerate(result01):
            ax2.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                     label=data["data error type"].to_numpy()[0], linestyle=linestyles[s],
                     c=colors[i])
            ax2.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         yerr=np.concatenate([np.array([0.02]),error01[i]["RMSE"].to_numpy()]),
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)
    ax2.plot(np.array([0]), ape, c="k")
    ax2.errorbar(np.array([0]), ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax2.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result01[0]["Nd"].to_numpy()
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels([f"{x}" for x in xticklabels])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("RMSE", fontsize=16)
    #ax2.set_xlabel(r"$N_{d}$", fontsize=16)
    ax2.xaxis.set_ticklabels([])
    ax2.set_ylim(0,0.2)
    ax2.xaxis.label.set_size(16)
    ax2.set_title(f"{model} model", fontsize=16)
    #ax2.legend()

    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=False, left=True, right=False)


    model = "GaussianSP"
    ax3 = plt.subplot(gs[3,0])
    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == nreplicas)
        & ((df["stat_model"] == model))
        )]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    linestyles = ["solid", "dashdot"]
    for s,swap_every in enumerate([0, 500]):
        _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
            & (df["nreplica"] == nreplicas) & (df["lambda_swap_every"] == swap_every)
            & ((df["stat_model"] == model))
            )]
        _df = _df.iloc[np.where(_df["uncertainties"] == "single")]
        result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
        error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
        result = result.reset_index()
        error = error.reset_index()
        result = [x for _, x in result.groupby(['data error type', 'prior error'])]
        result[-2], result[-1] = result[-1], result[-2]
        result01 = [_df for _df in result if _df["prior error"].to_numpy()[0] == prior_error]
        error = [x for _, x in error.groupby(['data error type', 'prior error'])]
        error[-2], error[-1] = error[-1], error[-2]
        error01 = [_df for _df in error if _df["prior error"].to_numpy()[0] == prior_error]

        colors = ["grey", "orange", "g", "r"]

        for i,data in enumerate(result01):
            ax3.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                     label=data["data error type"].to_numpy()[0], linestyle=linestyles[s],
                     c=colors[i])
            ax3.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         yerr=np.concatenate([np.array([0.02]),error01[i]["RMSE"].to_numpy()]),
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)
    ax3.plot(np.array([0]), ape, c="k")
    ax3.errorbar(np.array([0]), ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax3.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result01[0]["Nd"].to_numpy()
    ax3.set_xticks(xticklabels)
    ax3.set_xticklabels([f"{x}" for x in xticklabels])

    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_ylabel("RMSE", fontsize=16)
    #ax3.set_xlabel(r"$N_{d}$", fontsize=16)
    ax3.xaxis.set_ticklabels([])
    ax3.set_ylim(0,0.2)
    ax3.xaxis.label.set_size(16)
    ax3.set_title(f"{model} model", fontsize=16)
    #ax3.legend()

    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=False, left=True, right=False)

    model = "Outliers"
    ax4 = plt.subplot(gs[4,0])
    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == nreplicas)
        & ((df["stat_model"] == model))
        )]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    linestyles = ["solid", "dashdot"]
    for s,swap_every in enumerate([0, 500]):
        _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
            & (df["nreplica"] == nreplicas) & (df["lambda_swap_every"] == swap_every)
            & ((df["stat_model"] == model))
            )]
        _df = _df.iloc[np.where(_df["uncertainties"] == "single")]
        result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
        error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
        result = result.reset_index()
        error = error.reset_index()
        result = [x for _, x in result.groupby(['data error type', 'prior error'])]
        result[-2], result[-1] = result[-1], result[-2]
        result01 = [_df for _df in result if _df["prior error"].to_numpy()[0] == prior_error]
        error = [x for _, x in error.groupby(['data error type', 'prior error'])]
        error[-2], error[-1] = error[-1], error[-2]
        error01 = [_df for _df in error if _df["prior error"].to_numpy()[0] == prior_error]

        colors = ["grey", "orange", "g", "r"]

        for i,data in enumerate(result01):
            ax4.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                     label=data["data error type"].to_numpy()[0], linestyle=linestyles[s],
                     c=colors[i])
            ax4.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         yerr=np.concatenate([np.array([0.02]),error01[i]["RMSE"].to_numpy()]),
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)
    ax4.plot(np.array([0]), ape, c="k")
    ax4.errorbar(np.array([0]), ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax4.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result01[0]["Nd"].to_numpy()
    ax4.set_xticks(xticklabels)
    ax4.set_xticklabels([f"{x}" for x in xticklabels])

    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.set_ylabel("RMSE", fontsize=16)
    ax4.set_xlabel(r"$N_{d}$", fontsize=16)
    ax4.set_ylim(0,0.2)
    ax4.xaxis.label.set_size(16)
    ax4.set_title(f"{model} model", fontsize=16)
    ax4.legend()

    ax4.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=False, left=True, right=False)



    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),
            ax4.get_xticklabels(),
            ax4.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    fig.savefig(figname, dpi=800)
    #fig.show()
# }}}

# plot_model_comparison_RMSE:{{{
def plot_model_comparison_RMSE(df, figname, nstates):

    # PLOT
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
        & ((df["stat_model"] == "Outliers"))
        )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "g", "r"]

    for i,data in enumerate(result016):
        ax.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                 np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                     yerr=np.concatenate([np.array([0.06]),error016[i]["RMSE"].to_numpy()]),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    ax.plot(np.array([0]),ape, c="k")
    ax.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Outliers model\nPrior error = 0.16", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    # Add plot on y-axis
    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "Gaussian"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "multiple")]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()
    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-1], result[-2], result[-3], result[-4] = result[-2], result[-1], result[-3], result[-4]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-1], error[-2], error[-3], error[-4] = error[-2], error[-1], error[-3], error[-4]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008


    for i,data in enumerate(result016):
        ax1.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                 np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax1.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                     yerr=np.concatenate([np.array([0.06]),error016[i]["RMSE"].to_numpy()]),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    ax1.plot(np.array([0]),ape, c="k")
    ax1.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{d}$", fontsize=16)
    ax1.set_ylim(0,0.3)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Multiple sigmas\nPrior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    ax2 = plt.subplot(gs[1,0])

    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "OutliersSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        ax2.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                 np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax2.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                     yerr=np.concatenate([np.array([0.06]),error016[i]["RMSE"].to_numpy()]),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    ax2.plot(np.array([0]),ape, c="k")
    ax2.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels([f"{x}" for x in xticklabels])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("RMSE", fontsize=16)
    ax2.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax2.xaxis.label.set_size(16)
    ax2.set_title("Outliers SP model\nPrior error = 0.16", fontsize=16)
    ax2.legend()

    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    divider = make_axes_locatable(ax2,)
    # Add plot on y-axis
    ax3 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax2)



    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "GaussianSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        ax3.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                 np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax3.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                     np.concatenate([np.array([0.16]),data["RMSE"].to_numpy()]),
                     yerr=np.concatenate([np.array([0.06]),error016[i]["RMSE"].to_numpy()]),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    ax3.plot(np.array([0]),ape, c="k")
    ax3.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax3.set_xticks(xticklabels)
    ax3.set_xticklabels([f"{x}" for x in xticklabels])

    ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax3.set_ylim(0,0.3)
    ax3.xaxis.label.set_size(16)
    ax3.set_title("Gaussian SP model\nPrior error = 0.16", fontsize=16)
    ax3.legend()

    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figname)
    fig.show()
# }}}

# plot_model_comparison_biceps_score:{{{

def plot_model_comparison_biceps_score(df, figname, nstates):

    #print(df)
    cols = [i for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    cols = df.columns[cols].to_list()
    scores_cols,std_cols = cols[::2],cols[1::2]

    # PLOT
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
        & ((df["stat_model"] == "Outliers"))
        )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "g", "r"]

    for i,data in enumerate(result016):
        # FIXME
        ax.plot(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax.errorbar(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("BICePs Scores", fontsize=16)
    ax.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Outliers model\nPrior error = 0.16", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    # Add plot on y-axis
    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "Gaussian"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "multiple")]
    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-1], result[-2], result[-3], result[-4] = result[-2], result[-1], result[-3], result[-4]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-1], error[-2], error[-3], error[-4] = error[-2], error[-1], error[-3], error[-4]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008


    for i,data in enumerate(result016):
        # FIXME
        ax1.plot(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax1.errorbar(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax1.set_ylim(0,0.3)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Multiple sigmas\nPrior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    ax2 = plt.subplot(gs[1,0])

    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "OutliersSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]


    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        # FIXME
        ax2.plot(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax2.errorbar(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels([f"{x}" for x in xticklabels])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("BICePs Score", fontsize=16)
    ax2.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax2.xaxis.label.set_size(16)
    ax2.set_title("Outliers SP model\nPrior error = 0.16", fontsize=16)
    ax2.legend()

    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax2,)
    # Add plot on y-axis
    ax3 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax2)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "GaussianSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]


    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        # FIXME
        ax3.plot(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax3.errorbar(data["Nd"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax3.set_xticks(xticklabels)
    ax3.set_xticklabels([f"{x}" for x in xticklabels])

    ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    #ax3.set_ylim(0,0.3)
    ax3.xaxis.label.set_size(16)
    ax3.set_title("Gaussian SP model\nPrior error = 0.16", fontsize=16)
    ax3.legend()

    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figname)
    fig.show()


def hold_onto_this():
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "r", "g"]
    if result008 != []:
        for i,data in enumerate(result008):
            ax.plot(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])
            ax.errorbar(x=data["nreplica"].to_numpy(), y=data["score"].to_numpy(),
                        yerr=error008[i]["score"].to_numpy(), color=colors[i],
                        ls="None", fmt='o', capsize=3)

        #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
        #ax.set_yticklabels(yticklabels, rotation=15)
        xticklabels = result008[0]["nreplica"].to_numpy()
        ax.set_xticks(xticklabels)
        ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("BICePs Score", fontsize=16)
    ax.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Prior error = 0.08", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)

    for i,data in enumerate(result016):
        ax1.plot(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                 label=data["data error type"].to_numpy()[0], c=colors[i])
        ax1.errorbar(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                     yerr=error016[i]["score"].to_numpy(), color=colors[i],
                     ls="None", fmt='o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.set_ylim(-12,4.)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Prior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("figures/error_analysis_vary_replicas_biceps_scores.png")
    fig.show()

# }}}

# plot_model_comparison_bic_score:{{{

def plot_model_comparison_bic_score(df, figname, nstates):

    cols = [i for i,x in enumerate(df.columns.to_list()) if "BIC" in x if "Score" not in x]
    cols = np.sort(df.columns[cols].to_list())
    scores_cols = cols


    # PLOT
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
        & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
        & ((df["stat_model"] == "Outliers"))
        )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "g", "r"]

    for i,data in enumerate(result016):
        vals = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #bic_score = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #error016[i][scores_cols[-1]]
        #vals,uncert = bic_score.nominal, bic_score.std
        #exit()
        #for i in range(len(cols)):
        #    -0.5*(scores_cols[i]-scores_cols[0])
        #exit()

        # FIXME
        ax.plot(data["Nd"].to_numpy(), vals,
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        #ax.errorbar(data["Nd"].to_numpy(), vals,
        #             #yerr=error016[i][std_cols[-1]].to_numpy(),
        #             yerr=error016[i][scores_cols[-1]].to_numpy(),
        #             color=colors[i],
        #             ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("BIC", fontsize=16)
    ax.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Outliers model\nPrior error = 0.16", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    # Add plot on y-axis
    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "Gaussian"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "multiple")]
    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-1], result[-2], result[-3], result[-4] = result[-2], result[-1], result[-3], result[-4]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-1], error[-2], error[-3], error[-4] = error[-2], error[-1], error[-3], error[-4]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008


    for i,data in enumerate(result016):
        vals = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #bic_score = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #error016[i][scores_cols[-1]]
        #vals,uncert = bic_score.nominal, bic_score.std
        #exit()
        #for i in range(len(cols)):
        #    -0.5*(scores_cols[i]-scores_cols[0])
        #exit()

        # FIXME
        ax1.plot(data["Nd"].to_numpy(), vals,
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        #ax.errorbar(data["Nd"].to_numpy(), vals,
        #             #yerr=error016[i][std_cols[-1]].to_numpy(),
        #             yerr=error016[i][scores_cols[-1]].to_numpy(),
        #             color=colors[i],
        #             ls="None", fmt='-o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax1.set_ylim(0,0.3)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Multiple sigmas\nPrior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    ax2 = plt.subplot(gs[1,0])

    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "OutliersSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]


    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        vals = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #bic_score = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #error016[i][scores_cols[-1]]
        #vals,uncert = bic_score.nominal, bic_score.std
        #exit()
        #for i in range(len(cols)):
        #    -0.5*(scores_cols[i]-scores_cols[0])
        #exit()

        # FIXME
        ax2.plot(data["Nd"].to_numpy(), vals,
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        #ax.errorbar(data["Nd"].to_numpy(), vals,
        #             #yerr=error016[i][std_cols[-1]].to_numpy(),
        #             yerr=error016[i][scores_cols[-1]].to_numpy(),
        #             color=colors[i],
        #             ls="None", fmt='-o', capsize=3)

    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels([f"{x}" for x in xticklabels])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("BIC", fontsize=16)
    ax2.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax2.xaxis.label.set_size(16)
    ax2.set_title("Outliers SP model\nPrior error = 0.16", fontsize=16)
    ax2.legend()

    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax2,)
    # Add plot on y-axis
    ax3 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax2)


    _df = df.iloc[np.where((df["nstates"] == nstates) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "GaussianSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]


    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        vals = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #bic_score = -0.5*(data[scores_cols[-1]].to_numpy()-data[scores_cols[0]].to_numpy())
        #error016[i][scores_cols[-1]]
        #vals,uncert = bic_score.nominal, bic_score.std
        #exit()
        #for i in range(len(cols)):
        #    -0.5*(scores_cols[i]-scores_cols[0])
        #exit()

        # FIXME
        ax3.plot(data["Nd"].to_numpy(), vals,
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        #ax.errorbar(data["Nd"].to_numpy(), vals,
        #             #yerr=error016[i][std_cols[-1]].to_numpy(),
        #             yerr=error016[i][scores_cols[-1]].to_numpy(),
        #             color=colors[i],
        #             ls="None", fmt='-o', capsize=3)


    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["Nd"].to_numpy()
    ax3.set_xticks(xticklabels)
    ax3.set_xticklabels([f"{x}" for x in xticklabels])

    ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    #ax3.set_ylim(0,0.3)
    ax3.xaxis.label.set_size(16)
    ax3.set_title("Gaussian SP model\nPrior error = 0.16", fontsize=16)
    ax3.legend()

    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figname)
    fig.show()


def hold_onto_this():
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "r", "g"]
    if result008 != []:
        for i,data in enumerate(result008):
            ax.plot(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])
            ax.errorbar(x=data["nreplica"].to_numpy(), y=data["score"].to_numpy(),
                        yerr=error008[i]["score"].to_numpy(), color=colors[i],
                        ls="None", fmt='o', capsize=3)

        #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
        #ax.set_yticklabels(yticklabels, rotation=15)
        xticklabels = result008[0]["nreplica"].to_numpy()
        ax.set_xticks(xticklabels)
        ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("BICePs Score", fontsize=16)
    ax.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Prior error = 0.08", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)

    for i,data in enumerate(result016):
        ax1.plot(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                 label=data["data error type"].to_numpy()[0], c=colors[i])
        ax1.errorbar(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                     yerr=error016[i]["score"].to_numpy(), color=colors[i],
                     ls="None", fmt='o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.set_ylim(-12,4.)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Prior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("figures/error_analysis_vary_replicas_biceps_scores.png")
    fig.show()

# }}}

# plot_model_comparison_Nr_RMSE:{{{

def plot_model_comparison_Nr_RMSE(df, figname, nstates):

    # PLOT
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                         & ((df["stat_model"] == "Outliers"))
                         )]


    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "g", "r"]

    for i,data in enumerate(result016):
        ax.plot(data["nreplica"].to_numpy(),data["RMSE"].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax.errorbar(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                     yerr=error016[i]["RMSE"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    #ax.plot(np.array([0]),ape, c="k")
    #ax.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Outliers model\nPrior error = 0.16", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    # Add plot on y-axis
    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)


    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "Gaussian"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "multiple")]
    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()
    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-1], result[-2], result[-3], result[-4] = result[-2], result[-1], result[-3], result[-4]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-1], error[-2], error[-3], error[-4] = error[-2], error[-1], error[-3], error[-4]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008


    for i,data in enumerate(result016):
        ax1.plot(data["nreplica"].to_numpy(),data["RMSE"].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax1.errorbar(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                     yerr=error016[i]["RMSE"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    #ax1.plot(np.array([0]),ape, c="k")
    #ax1.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.set_ylim(0,0.3)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Multiple sigmas\nPrior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    ax2 = plt.subplot(gs[1,0])

    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "OutliersSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        ax2.plot(data["nreplica"].to_numpy(),data["RMSE"].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax2.errorbar(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                     yerr=error016[i]["RMSE"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    #ax2.plot(np.array([0]),ape, c="k")
    #ax2.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels([f"{x}" for x in xticklabels])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("RMSE", fontsize=16)
    ax2.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax2.xaxis.label.set_size(16)
    ax2.set_title("Outliers SP model\nPrior error = 0.16", fontsize=16)
    ax2.legend()

    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    divider = make_axes_locatable(ax2,)
    # Add plot on y-axis
    ax3 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax2)



    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "GaussianSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    ape = _df["avg prior error"].mean()
    ape_std = _df["avg prior error"].std()

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        ax3.plot(data["nreplica"].to_numpy(),data["RMSE"].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax3.errorbar(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                     yerr=error016[i]["RMSE"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    #ax3.plot(np.array([0]),ape, c="k")
    #ax3.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax3.set_xticks(xticklabels)
    ax3.set_xticklabels([f"{x}" for x in xticklabels])

    ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax3.set_ylim(0,0.3)
    ax3.xaxis.label.set_size(16)
    ax3.set_title("Gaussian SP model\nPrior error = 0.16", fontsize=16)
    ax3.legend()

    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figname)
    fig.show()

# }}}

# plot_model_comparison_Nr_BS:{{{

def plot_model_comparison_Nr_BS(df, figname, nstates):

    cols = [i for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    cols = df.columns[cols].to_list()
    scores_cols,std_cols = cols[::2],cols[1::2]

    # PLOT
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)

    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                         & ((df["stat_model"] == "Outliers"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]

    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "g", "r"]

    for i,data in enumerate(result016):
        # FIXME
        ax.plot(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax.errorbar(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("BICePs Scores", fontsize=16)
    ax.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Outliers model\nPrior error = 0.16", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    # Add plot on y-axis
    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)


    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "Gaussian"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "multiple")]
    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-1], result[-2], result[-3], result[-4] = result[-2], result[-1], result[-3], result[-4]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-1], error[-2], error[-3], error[-4] = error[-2], error[-1], error[-3], error[-4]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008


    for i,data in enumerate(result016):
        # FIXME
        ax1.plot(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax1.errorbar(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax1.set_ylim(0,0.3)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Multiple sigmas\nPrior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)



    ax2 = plt.subplot(gs[1,0])

    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "OutliersSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]


    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        # FIXME
        ax2.plot(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax2.errorbar(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels([f"{x}" for x in xticklabels])

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("BICePs Score", fontsize=16)
    ax2.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax2.xaxis.label.set_size(16)
    ax2.set_title("Outliers SP model\nPrior error = 0.16", fontsize=16)
    ax2.legend()

    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax2,)
    # Add plot on y-axis
    ax3 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax2)


    _df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 8) | (df["nreplica"] == 16) | (df["nreplica"] == 32) | (df["nreplica"] == 64) | (df["nreplica"] == 128)
                         & (df["Nd"] == 20)
                         #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == "GaussianSP"))
                         )]

    _df = _df.iloc[np.where(_df["uncertainties"] == "single")]


    result = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = _df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")

    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result[-2], result[-1] = result[-1], result[-2]
    result008 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.08]
    result016 = [_df for _df in result if _df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error[-2], error[-1] = error[-1], error[-2]
    error008 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.08]
    error016 = [_df for _df in error if _df["prior error"].to_numpy()[0] == 0.16]
    #result008

    for i,data in enumerate(result016):
        # FIXME
        ax3.plot(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                 label=data["data error type"].to_numpy()[0],
                 c=colors[i])
        ax3.errorbar(data["nreplica"].to_numpy(), data[scores_cols[-1]].to_numpy(),
                     #yerr=error016[i][std_cols[-1]].to_numpy(),
                     yerr=error016[i][scores_cols[-1]].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
    #ax.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax3.set_xticks(xticklabels)
    ax3.set_xticklabels([f"{x}" for x in xticklabels])

    ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_xlabel(r"$N_{d}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    #ax3.set_ylim(0,0.3)
    ax3.xaxis.label.set_size(16)
    ax3.set_title("Gaussian SP model\nPrior error = 0.16", fontsize=16)
    ax3.legend()

    ax3.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figname)
    fig.show()


def hold_onto_this():
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "r", "g"]
    if result008 != []:
        for i,data in enumerate(result008):
            ax.plot(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])
            ax.errorbar(x=data["nreplica"].to_numpy(), y=data["score"].to_numpy(),
                        yerr=error008[i]["score"].to_numpy(), color=colors[i],
                        ls="None", fmt='o', capsize=3)

        #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
        #ax.set_yticklabels(yticklabels, rotation=15)
        xticklabels = result008[0]["nreplica"].to_numpy()
        ax.set_xticks(xticklabels)
        ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("BICePs Score", fontsize=16)
    ax.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Prior error = 0.08", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)

    for i,data in enumerate(result016):
        ax1.plot(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                 label=data["data error type"].to_numpy()[0], c=colors[i])
        ax1.errorbar(data["nreplica"].to_numpy(), data["score"].to_numpy(),
                     yerr=error016[i]["score"].to_numpy(), color=colors[i],
                     ls="None", fmt='o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.set_ylim(-12,4.)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Prior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("figures/error_analysis_vary_replicas_biceps_scores.png")
    fig.show()

# }}}

# plot_biceps_score_against_DKL_each_data_point:{{{
def plot_biceps_score_against_DKL_each_data_point(df, figname):

#    df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
#                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
#                          #& (df["data error type"] == "Random & Systematic")
#                          #& (df["data error type"] == "None")
#                          #& (df["data error type"] == "Systematic")
#                          & (df["data error type"] == "Random")
#                         )]

    cols = [i for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    cols = df.columns[cols].to_list()
    scores_cols,std_cols = cols[::2],cols[1::2]

    Nds = np.sort(list(set(df["Nd"].to_list())))
    stat_models = np.sort(list(set(df["stat_model"].to_list())))

    fig = plt.figure(figsize=(16, 16))
    r,c = len(Nds),len(stat_models)
    gs = gridspec.GridSpec(r, c)

    ax = []
    ax.append(fig.add_subplot(gs[0,0]))
    shared_x_index = 0 #ax[0]
    shared_y_index = 0 #ax[0]
    column = 0
    for k in range(c):
        if k != column:
            ax.append(fig.add_subplot(gs[0,k]))
            column += 1
            shared_x_index += 4
        for i in range(1,r):
            ax.append(fig.add_subplot(gs[i,k], sharex=ax[shared_x_index], sharey=ax[k]))

    colors = ["grey", "orange", "g", "r"]
    labels = [col.split("BICePs Score ")[-1] for col in scores_cols]

    ax_Idx = 0
    for i in range(c):
        for k in range(r):
            _df = df.iloc[np.where((df["Nd"] == Nds[k]))]
            _df = _df.iloc[np.where(_df["stat_model"] == stat_models[i])]
            for j,col in enumerate(scores_cols):
                _df.plot.scatter(x="D_KL", y=col, c=colors[j], ax=ax[ax_Idx], label=labels[j])
            if k == 0: ax[ax_Idx].set_title(f"{stat_models[i]}\nNd = {Nds[k]}", fontsize=16)
            else:      ax[ax_Idx].set_title(f"Nd = {Nds[k]}", fontsize=16)
            #ax[ax_Idx].set_xlabel(r"$D_{KL}$", fontsize=16)
            #ax[ax_Idx].set_yscale('log')
            ax[ax_Idx].set_xscale('log')
            ax[ax_Idx].set_ylabel("", fontsize=16)
            ax[ax_Idx].get_legend().remove()
            ax[ax_Idx].xaxis.label.set_size(16)
            ax_Idx += 1

    for i,axis in enumerate(ax):
        # if it is the last row, then ...
        if ((i+1)%r == 0) and (i != 0):
            axis.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_xlabel(r"$D_{KL}$", fontsize=16)

        # if it is the first column, then ...
        elif i in list(range(r)):
            axis.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_ylabel("BICePs Scores", fontsize=16)
        else:
            axis.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                             bottom=True, top=True, left=True, right=True)
        if i == (r-1):
            axis.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_ylabel("BICePs Scores", fontsize=16)

    ax[-1].legend(loc="best",framealpha=0.5, fontsize=14)


    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    fig.savefig(figname)


#:}}}

# plot_biceps_score_against_DKL:{{{
def plot_biceps_score_against_DKL(df, figname, model="Outliers"):

    # TODO: Include all models
    # TODO: Include error
    # TODO: Include data error type
    models = ["Outliers", "OutliersSP", "Gaussian", "GaussianSP"]

    df = df.iloc[np.where((df["nstates"] == 5) & (df["nlambda"] == 4)
                         & (df["nreplica"] == 128) #& (df["lambda_swap_every"] == 0) #500)
                          & ((df["stat_model"] == model))
                          #& (df["data error type"] == "Random & Systematic")
                          & (df["data error type"] == "None")
                         )]

    cols = [i for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    cols = df.columns[cols].to_list()
    scores_cols,std_cols = cols[::2],cols[1::2]


    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 2)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = [plt.subplot(gs[i,k]) for k in range(2) for i in range(2)]
    colors = ["grey", "orange", "g", "r"]
    labels = [col.split("BICePs Score ")[-1] for col in scores_cols]
    for i,col in enumerate(scores_cols):
        df.plot.scatter(x="D_KL", y=col, c=colors[i], ax=ax[0], label=labels[i])
    ax[0].legend(loc="best",framealpha=0.5)
    ax[0].set_xlabel(r"$D_{KL}$", fontsize=16)
    ax[0].set_ylabel("BICePs Scores", fontsize=16)
    ax[0].set_xscale('log')
    ax[0].xaxis.label.set_size(16)
    #ax.set_title("Title", fontsize=16)
    ax[0].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)
    #divider = make_axes_locatable(ax,)

    #ax1 = divider.append_axes('top', size='100%', pad="0%", sharex=ax)
    df["D_KL"].plot.hist(alpha=0.5, bins=30, edgecolor='black', linewidth=1.2, color="b", ax=ax[1])
    ax[1].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)
    ax[1].set_ylabel("$P(D_{KL})$", fontsize=18)
    #ax[1].set_xscale('log')

    #ax2 = divider.append_axes('right', size='100%', pad="5%", sharey=ax)
    labels = [col.split("BICePs Score ")[-1] for col in scores_cols]
    for i,col in enumerate(scores_cols):
        df.plot.scatter(x="avg prior error", y=col, c=colors[i], ax=ax[2], label=labels[i])
    ax[2].legend(loc="best",framealpha=0.5)
    ax[2].set_xlabel("Prior Error", fontsize=16)
    ax[2].set_ylabel("", fontsize=16)
    ax[2].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    #divider = make_axes_locatable(ax2)
    #ax3 = divider.append_axes('top', size='100%', pad="0%", sharex=ax2)
    df["avg prior error"].plot.hist(alpha=0.5, bins=30, edgecolor='black', linewidth=1.2, color="b", ax=ax[3])
    ax[3].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)
    ax[3].set_ylabel("$P(\sigma_{P(X)})$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figname)


#:}}}

# plot_biceps_score_against_prior_error_each_data_point:{{{
def plot_biceps_score_against_prior_error_each_data_point(df, figname=None,
        data_error_type="None", nstates=5, nreplica=128, lambda_swap_every=500):

    #df = df.iloc[np.where((df["nstates"] == nstates) #& (df["nlambda"] == 4)
    #                     & (df["nreplica"] == nreplica) & (df["lambda_swap_every"] == lambda_swap_every) #500)
    #                      & (df["data error type"] == data_error_type)
    #                      #& (df["data error type"] == "Systematic")
    #                      #& (df["data error type"] == "Random")
    #                     )]

    cols = [i for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    cols = df.columns[cols].to_list()
    scores_cols,std_cols = cols[::2],cols[1::2]

    Nds = np.sort(list(set(df["Nd"].to_list())))
    stat_models = np.sort(list(set(df["stat_model"].to_list())))

    fig = plt.figure(figsize=(16, 16))
    r,c = len(Nds),len(stat_models)
    gs = gridspec.GridSpec(r, c)

    ax = []
    ax.append(fig.add_subplot(gs[0,0]))
    shared_x_index = 0 #ax[0]
    shared_y_index = 0 #ax[0]
    column = 0
    for k in range(c):
        if k != column:
            ax.append(fig.add_subplot(gs[0,k]))
            column += 1
            shared_x_index += 4
        for i in range(1,r):
            ax.append(fig.add_subplot(gs[i,k], sharex=ax[shared_x_index], sharey=ax[k]))

    colors = ["grey", "orange", "g", "r"]
    labels = [col.split("BICePs Score ")[-1] for col in scores_cols]

    ax_Idx = 0
    for i in range(c):
        for k in range(r):
            _df = df.iloc[np.where((df["Nd"] == Nds[k]))]
            _df = _df.iloc[np.where(_df["stat_model"] == stat_models[i])]
            for j,col in enumerate(scores_cols):
                _df.plot.scatter(x="avg prior error", y=col, c=colors[j], ax=ax[ax_Idx], label=labels[j])
            if k == 0: ax[ax_Idx].set_title(f"{stat_models[i]}\nNd = {Nds[k]}", fontsize=16)
            else:      ax[ax_Idx].set_title(f"Nd = {Nds[k]}", fontsize=16)
            ax[ax_Idx].set_ylabel("", fontsize=16)
            ax[ax_Idx].get_legend().remove()
            ax[ax_Idx].xaxis.label.set_size(16)
            ax_Idx += 1

    for i,axis in enumerate(ax):
        # if it is the last row, then ...
        if ((i+1)%r == 0) and (i != 0):
            axis.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_xlabel(r"Avg. Prior Error", fontsize=16)

        # if it is the first column, then ...
        elif i in list(range(r)):
            axis.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_ylabel("BICePs Scores", fontsize=16)
        else:
            axis.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                             bottom=True, top=True, left=True, right=True)
        if i == (r-1):
            axis.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_ylabel("BICePs Scores", fontsize=16)

    ax[-1].legend(loc="best",framealpha=0.5, fontsize=14)


    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    if figname: fig.savefig(figname)


#:}}}

# plot_bic_score_against_prior_error_each_data_point:{{{
def plot_bic_score_against_prior_error_each_data_point(df, figname=None,
        data_error_type="None", nstates=5, nreplica=128, lambda_swap_every=500):

    df = df.iloc[np.where((df["nstates"] == nstates) #& (df["nlambda"] == 4)
                         & (df["nreplica"] == nreplica) & (df["lambda_swap_every"] == lambda_swap_every) #500)
                          & (df["data error type"] == data_error_type)
                          #& (df["data error type"] == "Systematic")
                          #& (df["data error type"] == "Random")
                         )]

    cols = [i for i,x in enumerate(df.columns.to_list()) if "BIC Score" in x]
    cols = df.columns[cols].to_list()
    #scores_cols,std_cols = cols[::2],cols[1::2]
    scores_cols,std_cols = cols,cols

    Nds = np.sort(list(set(df["Nd"].to_list())))
    stat_models = np.sort(list(set(df["stat_model"].to_list())))

    fig = plt.figure(figsize=(16, 16))
    r,c = len(Nds),len(stat_models)
    gs = gridspec.GridSpec(r, c)

    ax = []
    ax.append(fig.add_subplot(gs[0,0]))
    shared_x_index = 0 #ax[0]
    shared_y_index = 0 #ax[0]
    column = 0
    for k in range(c):
        if k != column:
            ax.append(fig.add_subplot(gs[0,k]))
            column += 1
            shared_x_index += 4
        for i in range(1,r):
            ax.append(fig.add_subplot(gs[i,k], sharex=ax[shared_x_index], sharey=ax[k]))

    colors = ["grey", "orange", "g", "r"]
    labels = [col.split("BIC Score ")[-1] for col in scores_cols]

    ax_Idx = 0
    for i in range(c):
        for k in range(r):
            _df = df.iloc[np.where((df["Nd"] == Nds[k]))]
            _df = _df.iloc[np.where(_df["stat_model"] == stat_models[i])]
            for j,col in enumerate(scores_cols):
                _df.plot.scatter(x="avg prior error", y=col, c=colors[j], ax=ax[ax_Idx], label=labels[j])
            if k == 0: ax[ax_Idx].set_title(f"{stat_models[i]}\nNd = {Nds[k]}", fontsize=16)
            else:      ax[ax_Idx].set_title(f"Nd = {Nds[k]}", fontsize=16)
            ax[ax_Idx].set_ylabel("", fontsize=16)
            ax[ax_Idx].get_legend().remove()
            ax[ax_Idx].xaxis.label.set_size(16)
            ax_Idx += 1

    for i,axis in enumerate(ax):
        # if it is the last row, then ...
        if ((i+1)%r == 0) and (i != 0):
            axis.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_xlabel(r"Avg. Prior Error", fontsize=16)

        # if it is the first column, then ...
        elif i in list(range(r)):
            axis.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_ylabel("BIC Scores", fontsize=16)
        else:
            axis.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
                             bottom=True, top=True, left=True, right=True)
        if i == (r-1):
            axis.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                             bottom=True, top=True, left=True, right=True)
            axis.set_ylabel("BIC Scores", fontsize=16)

    ax[-1].legend(loc="best",framealpha=0.5, fontsize=14)


    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    if figname: fig.savefig(figname)


#:}}}

# plot_vary_N_replicas:{{{
def plot_vary_N_replicas(df, figname):

    df = db.iloc[np.where((db["nstates"] == 5) & (db["nlambda"] == 2)
                         #& (db["nreplica"] == 128)
                         & (db["nreplica"] == 8) | (db["nreplica"] == 16) | (db["nreplica"] == 32) | (db["nreplica"] == 64) | (db["nreplica"] == 128)
                         & (db["Nd"] == 20)
                         )]


    result = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("mean")
    error = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "data error type", "prior error"]).agg("std")
    #result["# datapoints"] = [N[4] for N in result.index.to_numpy()]
    result = result.reset_index()
    error = error.reset_index()
    result = [x for _, x in result.groupby(['data error type', 'prior error'])]
    result008 = [df for df in result if df["prior error"].to_numpy()[0] == 0.08]
    result016 = [df for df in result if df["prior error"].to_numpy()[0] == 0.16]
    error = [x for _, x in error.groupby(['data error type', 'prior error'])]
    error008 = [df for df in error if df["prior error"].to_numpy()[0] == 0.08]
    error016 = [df for df in error if df["prior error"].to_numpy()[0] == 0.16]
    #result008

    # PLOT
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])
    colors = ["grey", "orange", "r", "g"]
    if result008 != []:
        for i,data in enumerate(result008):
            ax.plot(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])
            ax.errorbar(x=data["nreplica"].to_numpy(), y=data["RMSE"].to_numpy(),
                        yerr=error008[i]["RMSE"].to_numpy(), color=colors[i],
                        ls="None", fmt='o', capsize=3)

        #yticklabels = ["%.2f"%label for label in np.array(list(range(0, 5)))*0.1]
        #ax.set_yticklabels(yticklabels, rotation=15)
        xticklabels = result008[0]["nreplica"].to_numpy()
        ax.set_xticks(xticklabels)
        ax.set_xticklabels([f"{x}" for x in xticklabels])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_xlabel(r"$N_{r}$", fontsize=16)
    #ax.set_ylim(0,0.3)
    ax.xaxis.label.set_size(16)
    ax.set_title("Prior error = 0.08", fontsize=16)
    ax.legend()

    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    divider = make_axes_locatable(ax,)
    # Add plot on y-axis
    ax1 = divider.append_axes('right', size='100%',
            pad="5%", sharey=ax)

    for i,data in enumerate(result016):
        ax1.plot(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                 label=data["data error type"].to_numpy()[0], c=colors[i])
        ax1.errorbar(data["nreplica"].to_numpy(), data["RMSE"].to_numpy(),
                     yerr=error016[i]["RMSE"].to_numpy(), color=colors[i],
                     ls="None", fmt='o', capsize=3)

    #ax1.set_yticklabels(yticklabels, rotation=15)
    xticklabels = result016[0]["nreplica"].to_numpy()
    ax1.set_xticks(xticklabels)
    ax1.set_xticklabels([f"{x}" for x in xticklabels])
    ax1.set_ylabel("", fontsize=16)
    ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.set_ylim(0,0.2)
    ax1.xaxis.label.set_size(16)
    ax1.set_title("Prior error = 0.16", fontsize=16)
    ax1.legend()

    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    #fig.savefig("figures/error_analysis_vary_replicas.png")
    fig.savefig(figname)
    fig.show()
# }}}

# plot_restraint_intensity:{{{
def plot_restraint_intensity(df, figname=None, figsize=None):

    result = []
    for row in df.iterrows():
        res_dict = {}
        nreplica = row[1]["nreplica"]
        N2 = nreplica**2
        k = np.array(row[1]["k"])
        for i in range(len(k)):
            res_dict[f"k{i}"] = k[i]
        res_dict["N2"] = int(N2)
        result.append(res_dict)
    result = pd.DataFrame(result)


    # PLOT
    cols = [col for col in result.columns if col != "N2"]
    Rs = []
    #ax = result.plot.scatter(x="N2", y="k0")
    prediction = np.poly1d(np.polyfit(result["N2"], result["k0"], 1))
    #ax.plot(result["N2"], prediction(result["N2"]), "k--")
    R2 = np.corrcoef(result["k0"], prediction(result["N2"]))[0][1]**2
    Rs.append(R2)
    avg_predictions = []
    avg_predictions.append(prediction(result["N2"]))
    if cols[1:]:
        for c in cols[1:]:
            #result.plot.scatter(x="N2", y=c, ax=ax)
            prediction = np.poly1d(np.polyfit(result["N2"], result[c], 1))
            #ax.plot(result["N2"], prediction(result["N2"]), "k--")
            R2 = np.corrcoef(result[c], prediction(result["N2"]))[0][1]**2
            Rs.append(R2)
            avg_predictions.append(prediction(result["N2"]))
    avg_predictions = np.mean(avg_predictions, axis=0)
    std_dev = result.std(axis=1)
    #print(std_dev)
    ax = plt.subplot()
    ax.plot(result["N2"], avg_predictions, "k--")
    ax.errorbar(result["N2"], avg_predictions,
                 yerr=std_dev.to_numpy(), color="k",
                 ls="None", fmt='-o', capsize=4)

    ax.set_ylabel(r"k ($k_{b}$T)", fontsize=16)
    #ax.set_xlabel(r"$N$", fontsize=16)
    #xticklabels = result["N"]
    xticklabels = [result["N2"][0], result["N2"][3], result["N2"][4], result["N2"][5]]
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"${int(np.sqrt(x))}^2$" for x in xticklabels])
    ax.set_xlabel(r"$N^{2}$", fontsize=16)
    ax.annotate(r'$R^{2}$ = %0.6g'%(np.mean(Rs)),
                xy=(8, result["k0"].max()*0.9),
                xycoords='data', fontsize=16)
    ax.set_title("harmonic restraint intensity")
    ax.set_ylim(0, (result["k0"].max()+std_dev.to_numpy()[-1])*1.2)

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    if figsize: fig.set_size_inches(*figsize)
    else: fig.set_size_inches(8,6)
    fig.tight_layout()
    if figname: fig.savefig(figname)
    #fig.show()


#:}}}

# plot_convergence:{{{
def plot_convergence(df, figname=None):

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax1 = plt.subplot(gs[0,0])

    biceps_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    columns = ["nsteps","nreplica","nlambda"]
    pops = []
    for row in df.iterrows():
        info = {}
        for i,val in enumerate(row[1]["pops"]):
            info[f"{i}"] = val
        for col in columns:
            info[col] = row[1][col]
        info["RMSE"] = row[1]["RMSE"]
        for col in score_cols:
            info[col] = row[1][col]
        pops.append(info)
    pops = pd.DataFrame(pops)
    prior_pops = pd.DataFrame([{i:val for i,val in enumerate(df["prior pops"].to_numpy()[0])}])
    states = list(range(len(prior_pops.columns.to_list())))

    group = pops.groupby(columns).agg("min")
    group = pd.DataFrame(group, )
    group[[str(state) for state in states]].plot.bar(rot=20, legend=None, ax=ax1, color="blue", edgecolor="k", linewidth=2,)
    ax1.set_ylabel("Populations", fontsize=18)
    ax1.xaxis.set_visible(False)

    divider = make_axes_locatable(ax1,)
    # Add plot on y-axis
    ax2 = divider.append_axes('right', size='15%',
            pad="12%", sharey=ax1, xticklabels=[])
    prior_pops.plot.bar(ax=ax2,color="blue", edgecolor="k", linewidth=2, width=1, legend=False)
    ax2.xaxis.set_visible(False)

    # Add plot on x-axis
    ax3 = divider.append_axes('bottom', size='50%',
            pad="10%", sharex=ax1, xticklabels=[])

    group["RMSE"].plot(ax=ax3, color="blue")
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_ylim(0,0.2)
    ax3.xaxis.label.set_size(16)

    # Add plot on x-axis
    ax4 = divider.append_axes('top', size='50%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(score_cols):
        group[col].plot(ax=ax4, color=colors[i], label=col)
        for j in range(len(group[col].index.to_list())):
            ax4.errorbar(j, df[col].to_numpy()[j],
                         yerr=df[std_cols[i]].to_numpy()[j],
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)

    ax4.legend()

    ax4.set_ylabel("BICePs Score", fontsize=16)
    ax4.xaxis.set_visible(False)

    # Setting the ticks and tick marks
    ticks = [ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax4.get_xticklabels(),
            ax4.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig.tight_layout()
    if figname: fig.savefig(figname)
#:}}}

# plot_convergence_with_bic:{{{
def plot_convergence_with_bic(df, figname=None):

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax1 = plt.subplot(gs[0,0])

    bic_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BIC Score" in x]
    biceps_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    columns = ["nsteps","nreplica","nlambda"]
    pops = []
    for row in df.iterrows():
        info = {}
        for i,val in enumerate(row[1]["pops"]):
            info[f"{i}"] = val
        for col in columns:
            info[col] = row[1][col]
        info["RMSE"] = row[1]["RMSE"]
        for col in score_cols:
            info[col] = row[1][col]
        pops.append(info)
    pops = pd.DataFrame(pops)
    prior_pops = pd.DataFrame([{i:val for i,val in enumerate(df["prior pops"].to_numpy()[0])}])
    states = list(range(len(prior_pops.columns.to_list())))

    group = pops.groupby(columns).agg("min")
    group = pd.DataFrame(group, )
    group[[str(state) for state in states]].plot.bar(rot=20, legend=None, ax=ax1, color="blue", edgecolor="k", linewidth=2,)
    ax1.set_ylabel("Populations", fontsize=18)
    ax1.xaxis.set_visible(False)

    divider = make_axes_locatable(ax1,)
    # Add plot on y-axis
    ax2 = divider.append_axes('right', size='15%',
            pad="12%", sharey=ax1, xticklabels=[])
    prior_pops.plot.bar(ax=ax2,color="blue", edgecolor="k", linewidth=2, width=1, legend=False)
    ax2.xaxis.set_visible(False)

    # Add plot on x-axis
    ax3 = divider.append_axes('bottom', size='50%',
            pad="10%", sharex=ax1, xticklabels=[])

    group["RMSE"].plot(ax=ax3, color="blue")
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_ylim(0,0.2)
    ax3.xaxis.label.set_size(16)

    # Add plot on x-axis
    ax4 = divider.append_axes('top', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(score_cols):
        group[col].plot(ax=ax4, color=colors[i], label=col)
        for j in range(len(group[col].index.to_list())):
            ax4.errorbar(j, df[col].to_numpy()[j],
                         yerr=df[std_cols[i]].to_numpy()[j],
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)

    ax4.legend()

    ax4.set_ylabel("BICePs\nScore", fontsize=16)
    ax4.xaxis.set_visible(False)

    # Add plot on x-axis
    ax5 = divider.append_axes('top', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(bic_score_cols):
        df[col].plot(ax=ax5, color=colors[i], label=col)
    ax5.legend()

    ax5.set_ylabel("BIC\nScore", fontsize=16)
    ax5.xaxis.set_visible(False)

    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_ylim(0,0.2)
    ax3.xaxis.label.set_size(16)


    # Setting the ticks and tick marks
    ticks = [ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks(),
             ax5.xaxis.get_minor_ticks(),
             ax5.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax4.get_xticklabels(),
            ax4.get_yticklabels(),
            ax5.get_xticklabels(),
            ax5.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig.tight_layout()
    if figname: fig.savefig(figname)
#:}}}

# plot_convergence_with_bic:{{{
def plot_convergence_with_BIC(df, figsize=(8, 6), figname=None):

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax1 = plt.subplot(gs[0,0])

    bic_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BIC Score" in x]
    biceps_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    columns = ["nsteps","nreplica","nlambda"]
    pops = []
    for row in df.iterrows():
        info = {}
        for i,val in enumerate(row[1]["pops"]):
            info[f"{i}"] = val
        for col in columns:
            info[col] = row[1][col]
        info["RMSE"] = row[1]["RMSE"]
        for col in score_cols:
            info[col] = row[1][col]
        pops.append(info)
    pops = pd.DataFrame(pops)
    prior_pops = pd.DataFrame([{i:val for i,val in enumerate(df["prior pops"].to_numpy()[0])}])
    states = list(range(len(prior_pops.columns.to_list())))

    group = pops.groupby(columns).agg("min")
    group = pd.DataFrame(group, )
    #group[[str(state) for state in states]].plot.bar(rot=20, legend=None, ax=ax1, color="blue", edgecolor="k", linewidth=2,)
    #ax1.set_ylabel("Populations", fontsize=18)

    group["RMSE"].plot(color="blue")
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
    #ax1.xaxis.set_visible(False)
    ax1.set_ylabel("RMSE", fontsize=16)
    ax1.set_ylim(0,0.2)
    ax1.xaxis.label.set_size(16)


    divider = make_axes_locatable(ax1,)
    ## Add plot on y-axis
    #ax2 = divider.append_axes('right', size='15%',
    #        pad="12%", sharey=ax1, xticklabels=[])
    #prior_pops.plot.bar(ax=ax2,color="blue", edgecolor="k", linewidth=2, width=1, legend=False)
    #ax2.xaxis.set_visible(False)

    ## Add plot on x-axis
    #ax3 = divider.append_axes('bottom', size='50%',
    #        pad="10%", sharex=ax1, xticklabels=[])

    #group["RMSE"].plot(ax=ax3, color="blue")
    #ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    #ax3.set_ylabel("RMSE", fontsize=16)
    #ax3.set_ylim(0,0.2)
    #ax3.xaxis.label.set_size(16)

    # Add plot on x-axis
    ax4 = divider.append_axes('top', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(score_cols):
        group[col].plot(ax=ax4, color=colors[i], label=col)
        for j in range(len(group[col].index.to_list())):
            ax4.errorbar(j, df[col].to_numpy()[j],
                         yerr=df[std_cols[i]].to_numpy()[j],
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)

    ax4.legend()

    ax4.set_ylabel("BICePs\nScore", fontsize=16)
    ax4.xaxis.set_visible(False)

    # Add plot on x-axis
    ax5 = divider.append_axes('top', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(bic_score_cols):
        df[col].plot(ax=ax5, color=colors[i], label=col)
    ax5.legend()

    ax5.set_ylabel("BIC\nScore", fontsize=16)
    ax5.xaxis.set_visible(False)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
    ax1.set_ylabel("RMSE", fontsize=16)
    ax1.set_ylim(0,0.2)
    ax1.xaxis.label.set_size(16)


    # Setting the ticks and tick marks
    ticks = [ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             #ax3.xaxis.get_minor_ticks(),
             #ax3.xaxis.get_major_ticks(),
             ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks(),
             ax5.xaxis.get_minor_ticks(),
             ax5.xaxis.get_major_ticks(),
             #ax2.xaxis.get_minor_ticks(),
             #ax2.xaxis.get_major_ticks()]
             ]
    marks = [ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax4.get_xticklabels(),
            ax4.get_yticklabels(),
            ax5.get_xticklabels(),
            ax5.get_yticklabels(),
            #ax3.get_xticklabels(),
            #ax3.get_yticklabels(),
            #ax2.get_xticklabels(),
            #ax2.get_yticklabels()]
            ]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig.tight_layout()
    if figname: fig.savefig(figname)
#:}}}

# plot_distributions:{{{
def plot_distributions(sampler, figname=None, xlim=(0,10), ylim=(0,8)):

    forward_model = sampler.get_forward_model()
    fm = pd.concat([forward_model[i]["model"] for i in range(len(forward_model))])
    #print(sampler.
    exit()

    npz = np.load(files[0], allow_pickle=True)["arr_0"].item()
    traj = npz["trajectory"]
    allowed_parameters = npz["allowed_parameters"]
    rest_type = npz["rest_type"]
    parameters = []
    for i in range(len(traj)):
        para_step = np.concatenate(traj[i][4])
        parameters.append({rest_type[k]: allowed_parameters[k][int(para_step[k])] for k in range(len(allowed_parameters))})
    df = pd.DataFrame(parameters)
    #print(df)

    files = biceps.toolbox.get_files(sem_file)
    data = {file.split("/")[-2]:pd.read_pickle(file) for file in files}
    column = [p for p in traj_files.split("/") if ("steps" and "replica") in p]
    data_df = pd.DataFrame(data[column[0]])
    error = df.drop(["gamma_noe"], axis=1)
    sig_r = np.sqrt(df**2 + data_df**2).drop(["gamma_noe"], axis=1)
    #new_df = 1/np.sqrt(df**2 + data_df**2)
    #new_df = new_df.drop(["gamma_noe"], axis=1)
    new_df = 1/sig_r**2

    ref = pd.DataFrame(npz["ref"][0]).transpose() # reference distribtuion
    print(new_df.mean())

    _model1 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files(datadir+"/*.noe")])
    #_model2 = pd.concat([pd.read_pickle(i) for i in biceps.toolbox.get_files(dir+"/*.noe")])
    ax = _model1["model"].plot.hist(alpha=1., bins=50, edgecolor='black', linewidth=1.2, density=1, color="b", figsize=(14, 6), label="data1")
    #_model2["model"].plot.hist(alpha=1., bins=50, edgecolor='black', linewidth=1.2, density=1, color="#800080", figsize=(14, 6), ax=ax, label="data2")
    error["sigma_noe"].plot.hist(color="#ffff00",alpha=1., bins=30, edgecolor='black', linewidth=1.2, density=1, ax=ax,figsize=(14, 6), )
    #sig_r["sigma_noe"].plot.hist(color=u'#ffe5b4',alpha=0.7, bins=30, edgecolor='black', linewidth=1.2, density=1, ax=ax,figsize=(14, 6),)
    new_df["sigma_noe"].plot.hist(color=u"#90ee90",alpha=0.6, bins=30, edgecolor='black', linewidth=1.5, density=1, ax=ax,figsize=(14, 6))
    #avg_dist.plot.hist(color='r',alpha=0.7, bins=30, edgecolor='black', linewidth=1.2, density=1, ax=ax,figsize=(14, 6),label="<r>")
    fm.plot.hist(color='r',alpha=0.7, bins=30, edgecolor='black', linewidth=1.2, density=1, ax=ax,figsize=(14, 6))
    #ref.plot.hist(color=u'#ffe5b4',alpha=0.65, bins=30, edgecolor='black', linewidth=1.2, density=1, ax=ax,figsize=(14, 6))
    ax.set_xlabel(r"", size=16)
    ax.set_ylabel(r"", size=16)
    ax.legend(["raw noe data",
               #"raw noe data (State B)",
               r"$P(\sigma^{B})$",
               #r"$P(\sigma_{r}) = \sqrt{(\sigma^{SEM})^{2}+(\sigma^{B})^{2}}}$",
               r"$P(k),\hspace{1}  k=\frac{1}{(\sigma^{SEM})^{2}+(\sigma^{B})^{2}}$",
               r"$P(f(X))$",
               #r"$P_{ref}(d_{j})$",
              ],
              prop={'size': 12})
    ax.set_ylim(ylim)

    # !!!!!!!!!!!!!!!!!!!!!! orange = #ff7f0e
    ax.set_xlim(xlim)
#:}}}

# plot biceps score against bic score:{{{

def plot_biceps_and_bic_with_RMSE(df, nstates=5, stat_model="Gaussian", vary="Nd", figname=None):
    #import matplotlib.gridspec as gridspec
    #from mpl_toolkits.axes_grid1 import make_axes_locatable

    #df = df.iloc[np.where(df["lambda_swap_every"] == 500)]
    #uncertainties = "multiple"
    #df = df.iloc[np.where(df["uncertainties"] == uncertainties)]
    #df = df.iloc[np.where(df["Nd"] < 25)]
    df = df.iloc[np.where(df["stat_model"] == stat_model)]

    results = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "prior error",'data error type']).agg("mean")
    results = results.reset_index()

    error = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "prior error",'data error type']).agg("std")
    error = error.reset_index()

    result016 = [x for _, x in results.groupby(['data error type'])]
    error016 = [x for _, x in error.groupby(['data error type'])]
    #result016 = [df for df in result if df["prior error"].to_numpy()[0] == 0.16]
    ape = df["avg prior error"].mean()
    ape_std = df["avg prior error"].std()

    ###############

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])

    ax.set_title(f"{nstates} states", fontsize=16)


    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        if vary == "Nd":
            for i,data in enumerate(result016):
                ax.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         label=data["data error type"].to_numpy()[0],
                         c=colors[i])
                ax.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                             np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                             yerr=np.concatenate([np.array([ape_std]),error016[i]["RMSE"].to_numpy()]),
                             color=colors[i],
                             ls="None", fmt='-o', capsize=3)

        else:
            for i,data in enumerate(result016):
                ax.plot(data[vary].to_numpy(), data["RMSE"].to_numpy(),
                        label=data["data error type"].to_numpy()[0], c=colors[i])

                ax.errorbar(data[vary].to_numpy(),
                         data["RMSE"].to_numpy(),
                         yerr=error016[i]["RMSE"].to_numpy(),
                         color=colors[i],
                         ls="None", fmt='-o', capsize=3)


    if vary == "Nd":
        ax.plot(np.array([0]),ape, c="k")
        ax.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_ylim(0, np.around(np.max(pd.concat(result016)["RMSE"].to_numpy())+0.1, decimals=1))
    ax.legend()
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ax1 = plt.subplot(gs[1,0], sharex=ax)
    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for i,data in enumerate(result016):
            ax1.plot(data[vary].to_numpy(), data["BICePs Score lam=1"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])

            ax1.errorbar(data[vary].to_numpy(),
                     data["BICePs Score lam=1"].to_numpy(),
                     yerr=error016[i]["BICePs Score lam=1"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)


    ax1.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylabel("BICePs Score", fontsize=16)
    #ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.xaxis.label.set_size(16)

    ax1.legend()


    ax2 = plt.subplot(gs[2,0], sharex=ax)

    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for i,data in enumerate(result016):
            ax2.plot(data[vary].to_numpy(), data["BIC Score lam=1"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])

            ax2.errorbar(data[vary].to_numpy(),
                     data["BIC Score lam=1"].to_numpy(),
                     yerr=error016[i]["BIC Score lam=1"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("BIC Score", fontsize=16)
    ax2.set_xlabel(vary, fontsize=16)
    ax2.legend()
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    xticklabels = np.sort(list(set(df[vary].to_numpy())))
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels(["%s"%x for x in xticklabels])

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
            ]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    if figname: fig.savefig(figname)
    return fig



#:}}}

# plot biceps score against bic score:{{{

def plot_biceps_and_BIC_with_RMSE(df, nstates=5, stat_model="Gaussian", vary="Nd", figname=None):
    #import matplotlib.gridspec as gridspec
    #from mpl_toolkits.axes_grid1 import make_axes_locatable

    #df = df.iloc[np.where(df["lambda_swap_every"] == 500)]
    #uncertainties = "multiple"
    #df = df.iloc[np.where(df["uncertainties"] == uncertainties)]
    #df = df.iloc[np.where(df["Nd"] < 25)]
    df = df.iloc[np.where(df["stat_model"] == stat_model)]

    results = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "prior error",'data error type']).agg("mean")
    results = results.reset_index()

    error = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "prior error",'data error type']).agg("std")
    error = error.reset_index()

    result016 = [x for _, x in results.groupby(['data error type'])]
    error016 = [x for _, x in error.groupby(['data error type'])]
    #result016 = [df for df in result if df["prior error"].to_numpy()[0] == 0.16]
    ape = df["avg prior error"].mean()
    ape_std = df["avg prior error"].std()

    ###############

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])

    ax.set_title(f"{nstates} states", fontsize=16)


    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        if vary == "Nd":
            for i,data in enumerate(result016):
                ax.plot(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                         np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                         label=data["data error type"].to_numpy()[0],
                         c=colors[i])
                ax.errorbar(np.concatenate([np.array([0]),data["Nd"].to_numpy()]),
                             np.concatenate([np.array([ape]),data["RMSE"].to_numpy()]),
                             yerr=np.concatenate([np.array([ape_std]),error016[i]["RMSE"].to_numpy()]),
                             color=colors[i],
                             ls="None", fmt='-o', capsize=3)

        else:
            for i,data in enumerate(result016):
                ax.plot(data[vary].to_numpy(), data["RMSE"].to_numpy(),
                        label=data["data error type"].to_numpy()[0], c=colors[i])

                ax.errorbar(data[vary].to_numpy(),
                         data["RMSE"].to_numpy(),
                         yerr=error016[i]["RMSE"].to_numpy(),
                         color=colors[i],
                         ls="None", fmt='-o', capsize=3)


    if vary == "Nd":
        ax.plot(np.array([0]),ape, c="k")
        ax.errorbar(np.array([0]),ape, yerr=ape_std, color="k",ls="None", fmt='-o', capsize=3)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_ylim(0, np.around(np.max(pd.concat(result016)["RMSE"].to_numpy())+0.1, decimals=1))
    ax.legend()
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    bic_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BIC Score" in x]
    biceps_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]

    ax1 = plt.subplot(gs[1,0], sharex=ax)
    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for k,col in enumerate(biceps_score_cols):
            for i,data in enumerate(result016):
                ax1.plot(data[vary].to_numpy(), data[col].to_numpy(),
                        label=data["data error type"].to_numpy()[0], c=colors[i])

                ax1.errorbar(data[vary].to_numpy(),
                         data[col].to_numpy(),
                         yerr=error016[i][col].to_numpy(),
                         color=colors[i],
                         ls="None", fmt='-o', capsize=3)


    ax1.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylabel("BICePs Score", fontsize=16)
    #ax1.set_xlabel(r"$N_{r}$", fontsize=16)
    ax1.xaxis.label.set_size(16)

    #ax1.legend()


    ax2 = plt.subplot(gs[2,0], sharex=ax)

    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for k,col in enumerate(bic_score_cols):
            for i,data in enumerate(result016):
                ax2.plot(data[vary].to_numpy(), data[col].to_numpy(),
                        label=data["data error type"].to_numpy()[0], c=colors[i])

                ax2.errorbar(data[vary].to_numpy(),
                         data[col].to_numpy(),
                         yerr=error016[i][col].to_numpy(),
                         color=colors[i],
                         ls="None", fmt='-o', capsize=3)

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("BIC Score", fontsize=16)
    ax2.set_xlabel(vary, fontsize=16)
    ax2.legend()
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    xticklabels = np.sort(list(set(df[vary].to_numpy())))
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels(["%s"%x for x in xticklabels])

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
            ]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    if figname: fig.savefig(figname)
    return fig



#:}}}

# plot_convergence_of_iterations:{{{
def plot_convergence_of_iterations(df, figname=None, use_index_as_iterations=True, figsize=(8, 6)):

    if use_index_as_iterations:
        df["iterations"] = df.index.to_list()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax1 = plt.subplot(gs[0,0])

    bic_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BIC Score" in x]
    biceps_score_cols = [x for i,x in enumerate(df.columns.to_list()) if "BICePs Score" in x]
    score_cols,std_cols = biceps_score_cols[::2],biceps_score_cols[1::2]
    columns = ["iterations","nsteps","nreplica","nlambda"]
    pops = []
    for row in df.iterrows():
        info = {}
        for i,val in enumerate(row[1]["pops"]):
            info[f"{i}"] = val
        for col in columns:
            info[col] = row[1][col]
        info["RMSE"] = row[1]["RMSE"]
        for col in score_cols:
            info[col] = row[1][col]
        pops.append(info)
    pops = pd.DataFrame(pops)
    prior_pops = pd.DataFrame([{i:val for i,val in enumerate(df["prior pops"].to_numpy()[0])}])
    states = list(range(len(prior_pops.columns.to_list())))

    group = pops.groupby(columns).agg("min")
    group = pd.DataFrame(group, )
    group[[str(state) for state in states]].plot.bar(rot=20, legend=None, ax=ax1, color="blue", edgecolor="k", linewidth=2,)
    ax1.set_ylabel("Populations", fontsize=18)
    ax1.xaxis.set_visible(False)

    divider = make_axes_locatable(ax1,)
    # Add plot on y-axis
    ax2 = divider.append_axes('right', size='15%',
            pad="12%", sharey=ax1, xticklabels=[])
    prior_pops.plot.bar(ax=ax2,color="blue", edgecolor="k", linewidth=2, width=1, legend=False)
    ax2.xaxis.set_visible(False)

    # Add plot on x-axis
    ax3 = divider.append_axes('bottom', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    group["RMSE"].plot(ax=ax3, color="blue")
    #ax3.tick_params(axis='both', which='major', labelsize=14)
    #ax3.set_xticklabels(ax3.get_xticklabels(), rotation=20)
    ax3.set_ylabel("RMSE", fontsize=16)
    ax3.set_ylim(0,0.2)
    #ax3.xaxis.label.set_size(16)

    # Add plot on x-axis
    ax6 = divider.append_axes('bottom', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    df["prior error"].plot(ax=ax6, color="blue")
    ax6.tick_params(axis='both', which='major', labelsize=14)
    ax6.set_xticklabels(ax3.get_xticklabels(), rotation=20)
    ax6.set_ylabel("prior error", fontsize=16)
    ax6.set_ylim(0,df["prior error"].to_numpy().max()+0.1)
    ax6.xaxis.label.set_size(16)
    ax6.set_xlabel("iterations", fontsize=16)


    # Add plot on x-axis
    ax4 = divider.append_axes('top', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(score_cols):
        group[col].plot(ax=ax4, color=colors[i], label=col)
        for j in range(len(group[col].index.to_list())):
            ax4.errorbar(j, df[col].to_numpy()[j],
                         yerr=df[std_cols[i]].to_numpy()[j],
                         color=colors[i], markersize=3,
                         ls="None", fmt='-o', capsize=3)

    ax4.legend()

    ax4.set_ylabel("BICePs\nScore", fontsize=16)
    ax4.xaxis.set_visible(False)

    # Add plot on x-axis
    ax5 = divider.append_axes('top', size='100%',
            pad="10%", sharex=ax1, xticklabels=[])

    colors = ["grey", "orange", "g", "r"]
    for i,col in enumerate(bic_score_cols):
        df[col].plot(ax=ax5, color=colors[i], label=col)
    ax5.legend()

    ax5.set_ylabel("BIC\nScore", fontsize=16)
    ax5.xaxis.set_visible(False)

    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    ax3.set_ylabel("RMSE", fontsize=16)
    #ax3.set_xlabel("iterations", fontsize=16)
    ax3.set_ylim(0,0.2)
    ax3.xaxis.label.set_size(16)


    # Setting the ticks and tick marks
    ticks = [ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
             ax3.xaxis.get_minor_ticks(),
             ax3.xaxis.get_major_ticks(),
             ax4.xaxis.get_minor_ticks(),
             ax4.xaxis.get_major_ticks(),
             ax5.xaxis.get_minor_ticks(),
             ax5.xaxis.get_major_ticks(),
             ax2.xaxis.get_minor_ticks(),
             ax2.xaxis.get_major_ticks()]
    marks = [ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ax4.get_xticklabels(),
            ax4.get_yticklabels(),
            ax5.get_xticklabels(),
            ax5.get_yticklabels(),
            ax3.get_xticklabels(),
            ax3.get_yticklabels(),
            ax2.get_xticklabels(),
            ax2.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig.tight_layout()
    if figname: fig.savefig(figname)
    return fig
#:}}}


def plot_memory_and_speed_performance(df, nstates=5, stat_model="Gaussian", vary="Nd", figname=None, figsize=(10, 8)):
    df = df.iloc[np.where(df["lambda_swap_every"] == 500)]
    df = df.iloc[np.where(df["stat_model"] == stat_model)]
    df["vmem"] = convert_to_Bytes(df["vmem"].to_numpy())/1e6
    df["swapmem"] = convert_to_Bytes(df["swapmem"].to_numpy())/1e6

    results = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "prior error",'data error type']).agg("mean")
    results = results.reset_index()

    error = df.groupby(["nsteps","nstates","nlambda","nreplica","Nd", "prior error",'data error type']).agg("std")
    error = error.reset_index()

    result016 = [x for _, x in results.groupby(['data error type'])]
    error016 = [x for _, x in error.groupby(['data error type'])]
    #result016 = [df for df in result if df["prior error"].to_numpy()[0] == 0.16]


    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1)#, width_ratios=[4, 1], wspace=0.001, hspace=0.5)
    ax = plt.subplot(gs[0,0])

    ax.set_title(f"{nstates} states", fontsize=16)


    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for i,data in enumerate(result016):
            ax.plot(data[vary].to_numpy(), data["RMSE"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])

            ax.errorbar(data[vary].to_numpy(),
                     data["RMSE"].to_numpy(),
                     yerr=error016[i]["RMSE"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_ylim(0, np.around(np.max(pd.concat(result016)["RMSE"].to_numpy())+0.1, decimals=1))
    ax.legend()
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)


    ax1 = plt.subplot(gs[1,0], sharex=ax)
    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for i,data in enumerate(result016):
            ax1.plot(data[vary].to_numpy(), data["time sampled (s)"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])

            ax1.errorbar(data[vary].to_numpy(),
                     data["time sampled (s)"].to_numpy(),
                     yerr=error016[i]["time sampled (s)"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)


    ax1.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylabel("Time Sampled (s)", fontsize=16)
    ax1.set_xlabel(vary, fontsize=16)
    ax1.xaxis.label.set_size(16)

    ax1.legend()


    ax2 = plt.subplot(gs[2,0], sharex=ax)

    colors = ["grey", "orange", "r", "g"]
    if result016 != []:
        for i,data in enumerate(result016):
            ax2.plot(data[vary].to_numpy(), data["vmem"].to_numpy(),
                    label=data["data error type"].to_numpy()[0], c=colors[i])
            #ax2.plot(data[vary].to_numpy(), data["swapmem"].to_numpy(),
            #        label=data["data error type"].to_numpy()[0], c=colors[i], linestyle="dotted")

            ax2.errorbar(data[vary].to_numpy(),
                     data["vmem"].to_numpy(),
                     yerr=error016[i]["vmem"].to_numpy(),
                     color=colors[i],
                     ls="None", fmt='-o', capsize=3)
            #ax2.errorbar(data[vary].to_numpy(),
            #         data["swapmem"].to_numpy(),
            #         yerr=error016[i]["swapmem"].to_numpy(),
            #         color=colors[i],
            #         ls="None", fmt='-o', capsize=3)

    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("Memory", fontsize=16)

    ax2.legend()
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                         bottom=True, top=True, left=True, right=True)

    xticklabels = np.sort(list(set(df[vary].to_numpy())))
    ax2.set_xticks(xticklabels)
    ax2.set_xticklabels(["%s"%x for x in xticklabels])

    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks(),
             ax1.xaxis.get_minor_ticks(),
             ax1.xaxis.get_major_ticks(),
            ]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels(),
            ax1.get_xticklabels(),
            ax1.get_yticklabels(),
            ]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig = ax.get_figure()
    fig.tight_layout()
    if figname: fig.savefig(figname)
    return fig





#:}}}



if __name__ == "__main__":


    dir="3_state/3_state_20_datapoints/Prior_error_0.16"
    dbName = dir+"/3-state_model.pkl"
    df = pd.read_pickle(dbName)
    #df = df.iloc[np.where(df["uncertainties"] == "single")]
    df = df.iloc[np.where(df["uncertainties"] == "multiple")]
    print(df)
    #plot_convergence(df, figname="3-state_model.png")
    plot_restraint_intensity(df, figname="restraint_intensity.png", figsize=(6,4))



if __name__ == "":


    outdir = "test_figures"
    biceps.toolbox.mkdir(outdir)


    db = pd.concat([
        pd.read_pickle("100_states.pkl"),
        pd.read_pickle("test1/100_states.pkl"),
        #pd.read_pickle("50_states.pkl"),
        #pd.read_pickle("test_50_states.pkl"),
        #pd.read_pickle("test_50_states_no_sigma_swap.pkl"),
        #pd.read_pickle("test1/test_50_states.pkl")
    ])
    db["prior error"] = db["prior error"].astype('float')
    db["avg prior error"] = db["avg prior error"].astype('float')

    nstates = 100 #50
    nreplicas = 16 #32 # 128
    figname = f"{outdir}/{nstates}_state_test_nreplicas_{nreplicas}.png"
    plot_all_models(db, figname, nstates=nstates, nreplicas=nreplicas)
    exit()


    db = pd.concat([
        #pd.read_pickle("test_N_replicas.pkl"),
        #pd.read_pickle("test1/test_N_replicas.pkl"),
        pd.read_pickle("test.pkl"),
        pd.read_pickle("test1/test.pkl")
    ])
    db["prior error"] = db["prior error"].astype('float')
    db["avg prior error"] = db["avg prior error"].astype('float')

#    # NOTE: Plot avg BIC against number of data points
#    figname = f"{outdir}/model_comparison_BIC_5_states.png"
#    plot_model_comparison_bic_score(db, figname, nstates=5)
#    exit()



    # NOTE: Plot biceps scrore against prior error for each data point
    figname = f"{outdir}/biceps_score_against_prior_error_each_datapoint.png"
    plot_biceps_score_against_prior_error_each_data_point(db, figname)
    # NOTE: Plot biceps scrore against D_KL for each data point
    figname = f"{outdir}/biceps_score_against_DKL_each_datapoint.png"
    plot_biceps_score_against_DKL_each_data_point(db, figname)

    exit()

    # NOTE: Plot biceps scrore against D_KL for each model of a given data error type
    models = ["Outliers", "OutliersSP", "Gaussian", "GaussianSP"]
    for model in models:
        figname = f"{outdir}/biceps_score_against_DKL_{model}.png"
        plot_biceps_score_against_DKL(db, figname, model)



    exit() #####################################################################

    db = pd.concat([
        pd.read_pickle("test_N_replicas.pkl"),
        pd.read_pickle("test1/test_N_replicas.pkl"),
        pd.read_pickle("test.pkl"),
        pd.read_pickle("test1/test.pkl")
    ])
    db["prior error"] = db["prior error"].astype('float')
    db["avg prior error"] = db["avg prior error"].astype('float')

    # NOTE: Plot vary N replica for all models for 5 states
    figname = f"{outdir}/model_comparison_Nr_RMSE_5_states.png"
    plot_model_comparison_Nr_RMSE(db, figname, nstates=5)

    figname = f"{outdir}/model_comparison_Nr_BICePs_Score_5_states.png"
    plot_model_comparison_Nr_BS(db, figname, nstates=5)

    exit() #####################################################################

    db = pd.concat([
        pd.read_pickle("test.pkl"),
        pd.read_pickle("test1/test.pkl")
    ])
    db["prior error"] = db["prior error"].astype('float')
    db["avg prior error"] = db["avg prior error"].astype('float')

    # NOTE: Plot model comparison for 5 states
    figname = f"{outdir}/model_comparison_RMSE_5_states.png"
    plot_model_comparison_RMSE(db, figname, nstates=5)

    figname = f"{outdir}/model_comparison_BICePs_Score_5_states.png"
    plot_model_comparison_biceps_score(db, figname, nstates=5)

    exit() #####################################################################

    # NOTE: Plot model comparison for 25 states
    db = pd.concat([
        pd.read_pickle("test_25_states.pkl"),
        pd.read_pickle("test1/test_25_states.pkl")
    ])
    db["prior error"] = db["prior error"].astype('float')
    db["avg prior error"] = db["avg prior error"].astype('float')

    figname = f"{outdir}/model_comparison_RMSE_25_states.png"
    plot_model_comparison_RMSE(db, figname, nstates=25)












