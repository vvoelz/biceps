# -*- coding: utf-8 -*-
import sys,os,math,string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .toolbox import format_label
from .toolbox import mpl_colors
import string


def single_exp_decay(x, a0, a1, tau1):
    """Function of a single exponential decay fitting.

    :math:`f(x) = a_{0} + a_{1}*exp(-(x/ \tau_{1}))`

    :param np.array x:
    :param float a0:
    :param float a1:
    :param float tau1:
    :return np.array: """

    return a0 + a1*np.exp(-(x/tau1))

def double_exp_decay(x, a0, a1, a2, tau1, tau2):
    """Function of a double exponential decay fitting.

    :math:`f(x) = a_{0} + a_{1}*exp(-(x/ \tau_{1})) + a_{2}*exp(-(x/ \tau_{2}))`

    :param np.array x:
    :param float a0:
    :param float a1:
    :param float a2:
    :param float tau1:
    :param float tau2:
    :return np.array: """

    return a0 + a1*np.exp(-(x/tau1)) + a2*np.exp(-(x/tau2))

def exponential_fit(autocorrelation, exp_function='single', v0=None, verbose=False):
    """Calls on :attr:`single_exp_decay` ('single') or :attr:`double_exp_decay`
    ('double') for an exponential fitting of an autocorrelation curve.
    See `SciPy curve fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    for more details.

    Args:
        autocorrelation(np.ndarray): the autocorrelation of some timeseries
        exp_function(str): default='single' ('single' or 'double'
        v0(list): Initial conditions for exponential fitting. Default for 'single' \
                is v0=[0.0, 1.0, 4000.]=[a0, a1, tau1] where :math:`a_{0} + a_{1}*exp(-(x/ \tau_{1}))` and\
                default for 'double' is v0=[0.0, 0.9, 0.1, 4000., 200.0]=[a0, a1, a2, tau1, tau2] where\
                :math:`f(x) = a_{0} + a_{1}*exp(-(x/ \tau_{1})) + a_{2}*exp(-(x/ \tau_{2}))`

    Returns:
        yfit(np.ndarray): the y-values of the fitted curve.

    """


    nsteps = autocorrelation.shape[0]
    if exp_function == 'single':
        if v0 is None: v0 = [0.0, 1.0 , 4000.]  # Initial guess [a0, a1, tau1] for a0 + a1*exp(-(x/tau1))
        popt, pcov = curve_fit(single_exp_decay, np.arange(nsteps), autocorrelation, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
        yFit_data = single_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2])
        if verbose: print(('Best-fit tau1 = %s +/- %s'%(popt[2],pcov[2][2])))
        max_tau = popt[2]
    else:
        if v0 is None: v0 = [0.0, 0.9, 0.1, 4000., 200.0]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
        popt, pcov = curve_fit(double_exp_decay, np.arange(nsteps), autocorrelation, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
        yFit_data = double_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2], popt[3], popt[4])
        if verbose: print(('Best-fit tau1 = %s +/- %s'%(popt[3],pcov[3][3])))
        if verbose: print(('Best-fit tau2 = %s +/- %s'%(popt[4],pcov[4][4])))
        #NOTE: This may need to be fixed later
        max_tau = max(popt[3],popt[4])
    return yFit_data,max_tau


def get_blocks(data, nblocks=5):
    """Method used to partition data into blocks. The data is a list of arrays,
    where each array is a separate time-series or autocorrelation.

    Args:
        data(list): list of separate timeseries

    """

    # slice the data into nblocks
    blocks = []
    for vec in data:
        dx = int(len(vec)/nblocks)
        blocks.append([vec[dx*n:dx*(n+1)] for n in range(nblocks)])
    return blocks


def compute_JSD(T1, T2, T_total, ind, allowed_parameters):
    """Compute JSD for a given part of trajectory.

    :math:`JSD = H(P_{comb}) - {\pi_{1}}{H(P_{1})} - {\pi_{2}}{H(P_{2})}`,
    where :math:`P_{comb}` is the combined data (:math:`P_{1} \cup P_{2}`).
    :math:`H` is the Shannon entropy of distribution :math:`P_{i}` and
    :math:`\pi_{i}` is the weight for the probability distribution :math:`P_{i}`.
    :math:`H(P_{i}) = \sum -\\frac{r_{i}}{N_{i}}*ln(\\frac{r_{i}}{N_{i}})`,
    where :math:`r_{i}` and :math:`N_{i}` represents sampled times of a
    specific parameter index and the total number of samples of the
    parameter, respectively

    :var T1, T2, T_total: part 1, part2 and total (part1 + part2)
    :var rest_type: experimental restraint type
    :var allowed_parameters: nuisacne parameters range
    :return float: Jensen–Shannon divergence
    """

    r1,r2,r_total = np.zeros(len(allowed_parameters)),np.zeros(len(allowed_parameters)),np.zeros(len(allowed_parameters))
    for frame in T1:
        parameter_indices = np.concatenate(frame[4])
        r1[parameter_indices[ind]]+=1
    for frame in T2:
        parameter_indices = np.concatenate(frame[4])
        r2[parameter_indices[ind]]+=1
    for frame in T_total:
        parameter_indices = np.concatenate(frame[4])
        r_total[parameter_indices[ind]]+=1
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
    return JSD




def g( f, max_tau = 10000, normalize=True):
    """Calculate the autocorrelaton function for a time-series f(t).

    :param np.array f:  a 1D numpy array containing the time series f(t)
    :param int max_tau: the maximum autocorrelation time to consider.
    :param bool normalize: if True, return g(tau)/g[0]
    :return np.array: a numpy array of size (max_tau+1,) containing g(tau)
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

def cal_auto(data, maxtau):
    """Calculates the autocorrelation"""

    autocorrs = []
    for timeseries in data:
        autocorrs.append( g(np.array(timeseries), max_tau=maxtau) )
    return autocorrs


def autocorrelation_time(autocorr):
    """Computes the autocorrelation time :math:`\\tau_{auto} = \int C_{\\tau} d\\tau`

    Args:
        autocorr(np.ndarray): an array containing the autocorrelation for \
                each restraint.
    """
    return autocorr.sum(axis=1)

def get_autocorrelation_time(data, method="auto", maxtau=5000):
    autocorr = cal_auto(data, maxtau)
    tau_c = autocorrelation_time(np.array(autocorr))
    return float(tau_c)


def exp_average(Et):
    result = 0.0
    for i in range(len(Et)): result += np.exp(-Et[i])
    result /= len(Et)
    return result




class Convergence(object):

    def __init__(self, traj=None, filename=None, outdir="./", verbose=False):
        """Convergence submodule for BICePs.

        Args:
            filename(str): relative path and filename to MCMC trajectory (NumPy npz file)
            outdir(str): relative path for output files
        """

        if (traj == None) and (filename == None):
            #raise()
            print("Must provide a trajectory or a relative path to a trajectory file.")
            exit()

        self.verbose = verbose
        if self.verbose: print(f'Loading {filename}...')
        if filename:
            self.traj = np.load(filename, allow_pickle=True)['arr_0'].item()
        if traj:
            if type(traj) == dict: self.traj = traj
            else: self.traj = traj.__dict__

        self.freq_save_traj = int(self.traj["trajectory"][1][0] - self.traj["trajectory"][0][0])
        if self.verbose: print('Collecting rest_type...')
        self.rest_type = self.traj['rest_type']
        if self.verbose: print('Collecting allowed_parameters...')
        self.allowed_parameters = self.traj['allowed_parameters']
        if self.verbose: print('Collecting sampled parameters...')
        #self.sampled_parameters = self.get_sampled_parameters()

        s = 0
        self.cols = []
        df = self.get_traces()
        _labels = self.get_labels()
        self.labels = []
        for i in range(len(self.rest_type)):
            df_array = df["%s"%(df.columns.to_list()[i])].to_numpy()
            if all(df_array == np.ones(df_array.shape)): continue
            if all(df_array == np.zeros(df_array.shape)): continue
            self.cols.append(self.rest_type[i])
            self.labels.append(_labels[i])
            s += 1
        n_columns = 1
        self.sampled_parameters = df[self.cols].to_numpy() # FIXME:
        self.sampled_parameters = self.sampled_parameters.T

        self.exp_function = "single"
        if outdir.endswith("/"): self.outdir = outdir
        else: self.outdir = outdir+"/"

    def get_labels(self):
        """Fetches the labels of each of the restraint types."""

        labels = []
        for i in range(len(self.rest_type)):
            if self.rest_type[i].count('_') == 0:
                if self.rest_type[i] == 'gamma':
                    labels.append('$\%s$'%self.rest_type[i])
                else:
                    labels.append('$%s$'%self.rest_type[i])
            elif self.rest_type[i].count('_') == 1:
                labels.append("$\%s_{%s}$"%(self.rest_type[i].split('_')[0],self.rest_type[i].split('_')[1]))
            elif self.rest_type[i].count('_') == 2:
                labels.append("$\%s_{{%s}_{%s}}$"%(self.rest_type[i].split('_')[0],self.rest_type[i].split('_')[1],self.rest_type[i].split('_')[2]))
        return labels


    def get_traces(self, traj_index=-1):

        npz = self.traj
        columns = self.rest_type #self.get_restraint_labels()
        df = pd.DataFrame(np.array(npz["traces"]).transpose(), columns)
        df = df.transpose()
        return df



    def plot_traces(self, figname="traj_traces.png", xlim=None, figsize=(14,6)):
        """Plot trajectory traces.

        Args:
            xlim(tuple): matplotlib x-axis limits
        """

        if self.verbose: print('Plotting traces...')
        total_steps = len(self.sampled_parameters[0])
        x = np.arange(1,total_steps+0.1,1)*self.freq_save_traj
        N_rest = len(self.cols) #len(self.traj["trajectory"][0][4])
        n_rows = 20
        if N_rest > n_rows:
            if self.verbose:
                print(f"Number of nuisance parameter traces: {N_rest}\n\
Too many plots for a single figure... only plotting the first {n_rows}.")
        else: n_rows = N_rest

        df = self.get_traces()
        n_columns = 1

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_columns)
        for i in range(len(self.cols)):
            #print(i)
            ax = plt.subplot(gs[(i,0)])
            ax.plot(x, df[self.cols[i]].to_numpy(), color='k',label=[self.cols[i]])
            ax.set_xlabel('', fontsize=18)
            ax.set_ylabel(format_label(self.cols[i]), fontsize=16)
            #ax.legend(loc='best')
            if xlim: ax.set_xlim(left=xlim[0], right=xlim[1])
            ax.tick_params(labelbottom=0, labeltop=0, labelleft=1,
                             labelright=0, bottom=1, top=1, left=1, right=0)
            ax.text(-0.12, 1.02, string.ascii_lowercase[i],
                    transform=ax.transAxes,size=20, weight='bold')

            if i == (n_rows-1): break

        ax.set_xlabel('steps', fontsize=18)
        ax.tick_params(labelbottom=1, labeltop=0, labelleft=1,
                         labelright=0, bottom=1, top=1, left=1, right=0)

        # Setting the ticks and tick marks
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

        fig.tight_layout()
        fig.savefig(self.outdir+figname)
        if self.verbose: print('Done!')
#        return fig





    def plot_auto_curve(self, xlim=None, figname="autocorrelation_curve.png",
            std_x=None, std_y=None, figsize=(12,10)):
        """Plot auto-correlation curve. This function saves a figure of
        auto-correlation with error bars at the 95% confidence interval
        (:math:`\\tau_{auto}` is rounded to the nearest integer).

        Args:
            xlim(tuple): matplotlib x-axis limits
            std_x(np.ndarray):
            std_y(np.ndarray):

        """

        if self.verbose: print('Plotting autocorrelation curve ...')
        #plt.figure( figsize=(3*len(self.rest_type),6))

        N_rest = len(self.traj["trajectory"][0][4])
        #print(N_rest)
        cols = []
        s = 0
        for i in range(len(self.autocorr)):
            if np.isnan(self.autocorr[i]).all(): continue
            else: cols.append(i); s+=1
        n_columns = 1

        n_rows = 20
        if s > n_rows:
            if self.verbose:
                print(f"Number of nuisance parameter traces: {s}\n\
Too many plots for a single figure... only plotting the first {n_rows}.")
        else: n_rows = s


        #figsize = (10*n_columns, n_rows)
        #figsize /= np.array(figsize).sum()
        #figsize *= 10
        #fig = plt.figure( figsize=figsize )
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_columns)
        pos = 0
        for k,i in enumerate(cols):
            #print(i)
            ax = plt.subplot(gs[(k,0)])
            ax.plot(np.arange(self.maxtau+1), self.autocorr[i], color='k', label=self.labels[i])
            ax.set_xlabel('', fontsize=18)
            ax.set_ylabel(r'$C_{\tau}$(%s)'%self.labels[i], fontsize=18)
            #ax.legend(loc='best')
            if xlim: ax.set_xlim(left=xlim[0], right=xlim[1])
            ax.tick_params(labelbottom=0, labeltop=0, labelleft=1,
                             labelright=0, bottom=1, top=1, left=1, right=0)

            try:
                j = int(round(self.tau_c[i]))
                ax.axvline(self.tau_c[i], color='k', linestyle="--")
            except(Exception) as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

            if (type(std_x) == np.ndarray) or (type(std_y) == np.ndarray):
                if (any(std_x) or any(std_y)) == np.nan: continue
                ax.annotate("$\\tau_{auto} = %i \\pm %i$"%(round(self.tau_c[i]),round(std_x[i])),
                        (self.tau_c[i], self.autocorr[i][j]),
                        xytext=(self.tau_c[i]+10, self.autocorr[i][j]+0.05))
                if (type(std_x) == np.ndarray):
                    ax.errorbar(self.tau_c[i], self.autocorr[i][j], xerr=std_x[i],
                            ecolor='k', fmt='o', capsize=10)

                if (type(std_y) == np.ndarray):
                    ax.fill_between(np.arange(self.maxtau+1),
                            self.autocorr[i]-std_y[i], self.autocorr[i]+std_y[i], color='r', alpha=0.4)


            else:
                try:
                    ax.annotate("$\\tau_{auto} = %i$"%(round(self.tau_c[i])),
                            (self.tau_c[i], self.autocorr[i][j]),
                            xytext=(self.tau_c[i]+10, self.autocorr[i][j]+0.05), fontsize=16)
                except(Exception) as e:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)


            ax.set_xlim(left=0)

            ax.set_xlabel('$\\tau$', fontsize=18)
            ax.tick_params(labelbottom=1, labeltop=0, labelleft=1,
                             labelright=0, bottom=1, top=1, left=1, right=0)

            # Setting the ticks and tick marks
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

            if xlim:
                ax.xlim(left=xlim[0], right=xlim[1])

            pos += 1
            if pos == (n_rows): break


        fig.tight_layout()
        fig.savefig(self.outdir+figname)
        if self.verbose: print('Done!')



    def plot_auto_curve_with_exp_fitting(self, figname="autocorrelation_curve_with_exp_fitting.png"):
        """Plot auto-correlation curve with an exponential fitting.

        :return figure: A figure of auto-correlation
        """

        if self.verbose: print('Plotting autocorrelation curve ...')
        plt.figure( figsize=(3*len(self.rest_type),6))
        for i in range(len(self.autocorr)):
            if len(self.rest_type) == 2:
                plt.subplot(len(self.autocorr),1,i+1)
            else:
                plt.subplot(len(self.autocorr),2,i+1)
            plt.plot(np.arange(self.maxtau+1), self.autocorr[i])
            plt.plot(np.arange(self.maxtau+1), self.yFits[i], 'r--')
            plt.xlabel('$\\tau$', fontsize=18)
            plt.ylabel('$C_{\\tau}$ for %s'%self.labels[i], fontsize=18)
            plt.xlim(left=0)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.outdir+figname)
        if self.verbose: print('Done!')


    def single_exp_decay(self, x, a0, a1, tau1):
        """Function of a single exponential decay fitting.

        :math:`f(x) = a_{0} + a_{1}*exp(-(x/\tau_{1}))`

        :param np.array x:
        :param float a0:
        :param float a1:
        :param float tau1:
        :return np.array: """

        return a0 + a1*np.exp(-(x/tau1))

    def double_exp_decay(self, x, a0, a1, a2, tau1, tau2):
        """Function of a double exponential decay fitting.

        :math:`f(x) = a_{0} + a_{1}*exp(-(x/\tau_{1})) + a_{2}*exp(-(x/\tau_{2}))`

        :param np.array x:
        :param float a0:
        :param float a1:
        :param float a2:
        :param float tau1:
        :param float tau2:
        :return np.array: """

        return a0 + a1*np.exp(-(x/tau1)) + a2*np.exp(-(x/tau2))

    def exponential_fit(self, ac, exp_function='single'):
        """Calls on :attr:`single_exp_decay` ('single') or :attr:`double_exp_decay`
        ('double') for an exponential fitting of an autocorrelation curve.
        See `SciPy curve fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
        for more details.

        :param np.ndarray ac:
        :param str default='single' exp_function:
        :return np.array yFit: the y-values of the fit curve."""

        nsteps = ac.shape[0]
        if exp_function == 'single':
            v0 = [0.0, 1.0 , 4000.]  # Initial guess [a0, a1, tau1] for a0 + a1*exp(-(x/tau1))
            popt, pcov = curve_fit(self.single_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
            yFit_data = self.single_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2])
            print(('Best-fit tau1 = %s +/- %s'%(popt[2],pcov[2][2])))
            max_tau = popt[2]
        else:
            v0 = [0.0, 0.9, 0.1, 4000., 200.0]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
            popt, pcov = curve_fit(self.double_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
            yFit_data = self.double_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2], popt[3], popt[4])
            print(('Best-fit tau1 = %s +/- %s'%(popt[3],pcov[3][3])))
            print(('Best-fit tau2 = %s +/- %s'%(popt[4],pcov[4][4])))
            #NOTE: This may need to be fixed later
            max_tau = max(popt[3],popt[4])
        return yFit_data,max_tau


    def get_autocorrelation_curves(self, method="auto", nblocks=5, maxtau=10000,
            plot_traces=False, figsize=(12,10)):
        """Compute autocorrelaton function for a time-series f(t), partition the
        data into the specified number of blocks and plot the autocorrelation curve.
        Saves a figure of autocorrelation curves for each restraint.

        Args:
            method(str): method for computing autocorrelation time; "block-avg-auto" or "exp" or "auto"
            nblocks(int): number of blocks to split up the trajectory
            maxtau(int): the upper bound of autocorrelation lag time
            plot_traces(bool): plot the trajectory traces?
        """

        self.maxtau = maxtau
        # Python
        if self.verbose: print('Calculating autocorrelation ...')
        self.autocorr = cal_auto(self.sampled_parameters, maxtau=maxtau)
        if self.verbose: print("Done!")
        if self.verbose: print('Calculating autocorrelation times...')
        self.tau_c = autocorrelation_time(np.array(self.autocorr))
        #print(self.tau_c)
        if self.verbose: print("Done!")

        if method in ["block-avg-auto","auto"]:
            if method == "auto":
                nblocks = 1
            blocks = get_blocks(self.sampled_parameters, nblocks)
            x,y = [],[]
            for i in range(len(blocks)):
                arr = np.array(cal_auto(blocks[i], maxtau=maxtau))
                y.append(arr)
                auto_corr = np.array(autocorrelation_time(arr))
                x.append(auto_corr)

            x, y = np.array(x),np.array(y)
            self.autocorr = np.average(y, axis=1)
            self.tau_c = np.average(x, axis=1)

            # Check to see if there are any negative autocorrelation times
            # NOTE: Should mention something about negative autocorrelation times
            if any(i < 0 for i in self.tau_c):
                print("NOTE: Found a negative autocorrelation time...")

            # If the number of blocks is 1, then we can't get std.
            if nblocks == 1:
                std_y,std_x = None,None
            else:
                std_y = np.std(y, axis=1)
                std_x = np.std(x, axis=1)
            self.plot_auto_curve(std_x=std_x, std_y=std_y, figsize=figsize)

        elif method == "exp":
            from scipy.optimize import curve_fit
            self.yFits,popts = [],[]
            for i in range(len(self.autocorr)):
                yFit,popt = self.exponential_fit(self.autocorr[i], exp_function=self.exp_function)
                popts.append(popt)
                self.yFits.append(yFit)
            self.tau_c = np.array(popts) #np.max(popts)
            print(("self.tau_c = %s"%self.tau_c))
            self.plot_auto_curve_with_exp_fitting()
        else:
            raise KeyError(f"Method must be 'block-avg-auto' or 'exp' or\
                    'auto'. You provided: {method}")

        if plot_traces:
            self.plot_traces()


    def process(self, nblock=5, nfold=10, nround=100, savefile=True,
            block_avg=False, normalize=True, plot=True):
        """Process the trajectory and execute :func:`compute_JSD` with
        :func:`plot_JSD_conv` and :func:`plot_JSD_distribution`.
        If :attr:`block_avg=True`, then block averaging will be executed and
        :func:`plot_block_avg` will be executed as well.

        Args:
             nblock(int): is the number of partitions in the time series
             nfold(int): is the number of partitions in the shuffled (subsampled) trajectory
             nround(int): is the number of rounds of bootstrapping when computing JSDs
             savefile(bool):
             block_avg(bool): use block averaging
             verbose(bool): verbosity
        """

        # Only do columns that are sigmas
        self._cols = [col for col in self.cols if "sigma" in col.lower()]
        print(self._cols)
        _tau_c = [self.tau_c[i] for i,col in enumerate(self.cols) if "sigma" in col.lower()]
        if block_avg:
            r_total = [[] for i in range(len(self._cols))]
            r_max = [[] for i in range(len(self._cols))]
            for i in range(len(self._cols)):
                k = int(np.where(self._cols[i] == np.array(self.rest_type))[0])
                allowed_parameters = self.allowed_parameters[k]

                tau_auto = _tau_c[i]
                tau = int(1+2*tau_auto)
                T_new = self.traj['trajectory'][::tau]
                nsnaps = len(T_new)
                dx = int(nsnaps/nfold)
                for subset in range(nblock):
                    T_total = T_new[dx*subset:dx*(subset+1)]
                    r_grid = np.zeros(len(allowed_parameters))
                    for k in T_total:
                        ind = np.concatenate(k[4])[i]
                        r_grid[ind]+=1
                    r_total[i].append(r_grid)
                    r_max[i].append(allowed_parameters[np.argmax(r_grid)])
            if plot: self.plot_block_avg(nblock,r_max)

        all_JSD=[[] for i in range(len(_tau_c))]      # create JSD list
        all_JSDs=[[[] for i in range(nfold)] for j in range(len(_tau_c))]   # create JSD list of distribution
        if self.verbose: print('Calculating JSDs ...')
        for i in range(len(self._cols)):
            k = int(np.where(self._cols[i] == np.array(self.rest_type))[0])
            allowed_parameters = self.allowed_parameters[k]

            tau_auto = _tau_c[i]
            if np.isnan(tau_auto):
                print(f"Unable to compute tau_auto for {self._cols[i]}")
                continue

            tau = int(1+2*tau_auto)
            T_new = self.traj['trajectory'][::tau]
            nsnaps = len(T_new)
            if nsnaps < 2*nfold:
                raise ValueError(f"Not enough data left after subsampling using auto-correlation time with the given nfold:\n"
                     f"nsnaps < 2*nfold = {nsnaps} < {2 * nfold}")
                exit()
            dx = int(nsnaps/nfold)
            for subset in range(nfold):
                half = int(dx * (subset+1)/2)
                T1 = T_new[:half]     # first half of the trajectory
                T2 = T_new[half:dx*(subset+1)]    # second half of the trajectory
                T_total = T_new[:dx*(subset+1)]     # total trajectory
                all_JSD[i].append(compute_JSD(T1,T2,T_total,k,allowed_parameters))   # compute JSD
                for r in range(nround):      # now let's mix this dataset
                    mT1 = np.random.choice(len(T_total),int(len(T_total)/2),replace=False)    # randomly pickup snapshots (index) as the first part
                    mT2 = np.delete(np.arange(0,len(T_total),1),mT1)           # take the rest (index) as the second part
                    temp_T1, temp_T2 = [],[]
                    for snapshot in mT1:
                        temp_T1.append(T_total[snapshot])      # take the first part dataset from the trajectory
                    for snapshot in mT2:
                        temp_T2.append(T_total[snapshot])      # take the second part dataset from the trajectory
                    all_JSDs[i][subset].append(compute_JSD(temp_T1,temp_T2,T_total,k,allowed_parameters))

        self.all_JSD, self.all_JSDs = np.array(all_JSD), np.array(all_JSDs)
        if savefile:
            np.save(os.path.join(self.outdir,"all_JSD.npy"), self.all_JSD)
            np.save(os.path.join(self.outdir,"all_JSDs.npy"), self.all_JSDs)
        if self.verbose: print('Done!')
        if plot: self.plot_JSD_distribution(self.all_JSD, self.all_JSDs, nround, nfold)
        if plot: self.plot_JSD_conv(self.all_JSD, self.all_JSDs)


    def plot_block_avg(self, nblock, r_max, figname="block_avg.png"):
        """Plot block average

        Args:
             nblock(int): is the number of partitions in the time series
             r_max(np.ndarray): maximum sampled parameters for each restraint
             figname(str): figure name without relative path (taken care of)
        """

        plt.figure(figsize=(10,5*len(self.rest_type)))
        x=np.arange(1.,nblock+1.,1.)
        #colors=['red','blue','black','green']
        colors=mpl_colors
        #print(len(self.rest_type))
        #print(len(r_max))
        #print(len(self.labels))
        for i in range(len(self.rest_type)):
            total_max = self.allowed_parameters[i][np.argmax(self.traj['sampled_parameters'][i])]
            plt.subplot(len(self.rest_type),1,i+1)
            plt.plot(x, r_max[i], 'o-', color=colors[i], label=self.labels[i])
            plt.xlabel('block', fontsize=18)
            plt.ylabel('allowed '+self.labels[i], fontsize=18)
            plt.ylim(min(self.allowed_parameters[i]),max(self.allowed_parameters[i]))
            plt.plot(nblock-0.2,total_max,'*',ms=20,color='green',label='total max')
            plt.legend(loc='best', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.outdir+figname)


    def plot_JSD_conv(self, JSD, JSDs, p_limit=0.99):
        """Plot Jensen–Shannon divergence (JSD) distribution for convergence check.

        :var all_JSD: JSDs for different amount of total dataset
        :var all_JSDs: JSDs for different amount of total dataset from bootstrapping
        :var rest_type: experimental restraint type
        :param float default=0.99 p_limit: plot a red horizontal dotted line at this y-value
        :return figure: A figure of JSD and JSDs distribution
        """

        if self.verbose: print('plotting JSDs ...')
        for k in range(len(JSD)):
            plt.figure(figsize=(10,5))
            for j in range(len(JSD[0])):
                plt.subplot(2,5,j+1)
                all_JSDs = np.append(JSDs[k][j],JSD[k][j])
                JSDs_sorted = np.sort(all_JSDs)
                p = np.arange(len(all_JSDs), dtype=float)
                ind = np.where(JSDs_sorted==JSD[k][j])[0][0]
                norm =  len(all_JSDs) - 1.
                # Red Dot
                plt.plot(JSDs_sorted[ind], p[ind]/norm,
                        'o',ms=5,color='red',label='%.3f'%(p[ind]/norm))
#                plt.annotate('%.3f'%(p[ind]/norm),xy=(JSDs_sorted[ind],p[ind]/norm),color='red',fontsize=6)
                # Blue Curve
                plt.plot(JSDs_sorted, p/norm)
                # Horizontal Line
                plt.plot([0.0, JSDs_sorted[ind]],
                        [p[ind]/norm, p[ind]/norm], 'k')
                # Red horizontal line
                plt.plot([0.0, np.max(JSDs_sorted)],
                        [p_limit, p_limit], '--r')

                # Vertical Line
                plt.plot([JSDs_sorted[ind], JSDs_sorted[ind]],
                        [0.0, p[ind]/norm], 'k')

                plt.ylim(bottom=0.0, top=1.0)
                plt.xlim(left=0.0)
                plt.xlabel('JSD')
                plt.ylabel('$p$')
                plt.xticks(fontsize=6)
                plt.locator_params(axis='x',nbins=5)
                plt.title('%d'%(10*(j+1))+'%',fontsize=10)
                plt.legend(loc='best',fontsize=6)
            plt.tight_layout()
            plt.savefig(self.outdir+'JSD_conv_%s.pdf'%self._cols[k])
        if self.verbose: print('Done')



    def plot_JSD_distribution(self, all_JSD, all_JSDs, nround, nfold,
                figname="JSD_distribution.png", label_size=16, figsize=(6,6)):
        """Plots the distributions for JSD"""

        # convert shape of all_JSD from (fold,n_rest) to (n_rest,fold)
        #n_rest = len(self.rest_type)
        n_rest = len(all_JSDs)
        # compute mean, std of JSDs from each fold dataset of each restraint
        JSD_dist = [[] for i in range(n_rest)]
        JSD_std = [[] for i in range(n_rest)]
        for rest in range(n_rest):
            for f in range(nfold):
                temp_JSD = []
                for r in range(nround):
                    temp_JSD.append(all_JSDs[rest][f][r])
                JSD_dist[rest].append(np.mean(temp_JSD))
                JSD_std[rest].append(np.std(temp_JSD))
        plt.figure(figsize=(8,5*n_rest))
        x=np.arange(int(100/nfold),101.,int(100/nfold))   # the dataset was divided into ten folds (this is the only hard coded part)
        for i in range(n_rest):
            plt.subplot(n_rest,1,i+1)
            plt.plot(x,all_JSD[i].transpose(), color=mpl_colors[i], label="__no_legend__")
            plt.scatter(x, all_JSD[i].transpose(), edgecolor='k', color=mpl_colors[i], label=self._cols[i])
            # at 95% confidence interval
            bounds = np.sort(all_JSDs[i])
            # remove top 50 and lower 50
            lower = bounds[:, int(nround*0.05)]
            upper = bounds[:, int(nround*0.95)]
            plt.fill_between(x,lower,upper,color=mpl_colors[i],alpha=0.2)
            plt.xlabel('dataset (%)', size=label_size)
            ylabel = r"%s}$"%self._cols[i].replace("sigma_", "$\sigma_{")
            plt.ylabel(f'JSD (%s)'%ylabel, size=label_size)
            plt.legend(loc='best', fontsize=label_size-2)
            plt.xticks(fontsize=label_size-2)
            plt.yticks(fontsize=label_size-2)
        plt.tight_layout()
        plt.savefig(self.outdir+figname)







