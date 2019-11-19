# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText
#import c_convergence as c_conv


class Convergence(object):
    """Convergence submodule for BICePs. """

    def __init__(self, trajfile=None, maxtau=10000, nblock=5,
            nfold=10, nround=1000):

        if trajfile is None:
            raise ValueError("Trajectory file is necessary")
        else:
            print('Loading trajectory file...')
#            self.traj = np.load(trajfile, allow_pickle=True)
            self.traj = np.load(trajfile)['arr_0'].item()
        print('Collecting rest_type...')
        self.rest_type = self.traj['rest_type']
        print('Collecting allowed_parameters...')
        self.allowed_parameters = self.traj['allowed_parameters']
        print('Collecting sampled parameters...')
        self.sampled_parameters = self.get_sampled_parameters()
        self.labels = self.get_labels()
        self.maxtau = maxtau
        self.nblock = nblock
        self.nfold = nfold
        self.nround = nround

    def get_sampled_parameters(self):
        """Get sampled parameters along time (steps).

        :param np.array traj: output trajectory from BICePs sampling
        :var default=None rest_type: experimental restraint type
        :return list: A list of all nuisance paramters sampled
        """

        parameters = []
        for i in range(len(self.rest_type)):
            parameters.append(np.array(self.traj['traces'])[:,i])
        parameters = np.array(parameters)
        return parameters

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


    def plot_traces(self, fname="traj_traces.png"):
        """Plot trajectory traces.

        :return figure: A figure
        """

        print('Plotting traces...')
        total_steps = len(self.sampled_parameters[0])
        x = np.arange(1,total_steps+0.1,1)
        n_rest = len(self.rest_type)
        plt.figure(figsize=(3*n_rest,15))

        for i in range(len(self.rest_type)):
            plt.subplot(len(self.rest_type), 1, i+1)
            plt.plot(x, self.sampled_parameters[i],label=self.labels[i])
            plt.ylabel(self.labels[i])
            plt.xlabel('steps')
            plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fname)
        print('Done!')

    def plot_auto_curve(self, autocorrs, tau_c, labels,
            fname="autocorrelation_curve.png", std_x=None, std_y=None):
        """Plot auto-correlation curve.

        :param autocorrs:
        :param tau_c:
        :param labels:
        :return figure: A figure of auto-correlation with error bars at the 95%
        confidence interval (tau0 is rounded to the nearest integer).
        """

        print('plotting autocorrelation curve ...')
        plt.figure( figsize=(3*len(self.rest_type),10))
        for i in range(len(autocorrs)):
            if len(self.rest_type) == 2:
                plt.subplot(len(autocorrs),1,i+1)
            else:
                plt.subplot(len(autocorrs),2,i+1)
            plt.plot(np.arange(self.maxtau+1), autocorrs[i])
            j = round(tau_c[i])
            plt.axvline(tau_c[i], color='k', linestyle="--")
            # If the number of blocks is 1, then we can't get std.
            if len(autocorrs) > 1:
                plt.annotate("$\\tau_{0} = %i \\pm %i$"%(round(tau_c[i]),round(std_x[i])),
                        (tau_c[i], autocorrs[i][j]),
                        xytext=(tau_c[i]+10, autocorrs[i][j]+0.05))
            else:
                plt.annotate("$\\tau_{0} = %i$"%(round(tau_c[i]),
                        (tau_c[i], autocorrs[i][j]),
                        xytext=(tau_c[i]+10, autocorrs[i][j]+0.05))

            if std_x != None:
                plt.errorbar(tau_c[i], autocorrs[i][j], xerr=std_x[i],
                        ecolor='k', fmt='o', capsize=10)

            if std_y != None:
                plt.fill_between(np.arange(self.maxtau+1),
                        autocorrs[i]-std_y[i], autocorrs[i]+std_y[i], color='r', alpha=0.4)

            plt.xlabel('$\\tau$')
            plt.ylabel('$g(\\tau)$ for %s'%labels[i])
            plt.xlim(left=0)
        plt.tight_layout()
        plt.savefig(fname)
        print('Done!')

    def plot_auto_curve_with_exp_fitting(self, autocorrs, yFits, labels,
            fname="autocorrelation_curve_with_exp_fitting.png"):
        """Plot auto-correlation curve.

        :param autocorrs:
        :param yFits:
        :param labels:
        :return figure: A figure of auto-correlation
        """

        print('plotting autocorrelation curve ...')
        plt.figure( figsize=(3*len(self.rest_type),10))
        for i in range(len(autocorrs)):
            plt.subplot(len(autocorrs),2,i+1)
            plt.plot(np.arange(self.maxtau+1), autocorrs[i])
            plt.plot(np.arange(self.maxtau+1), yFits[i], 'r--')
            plt.xlabel('$\\tau$')
            plt.ylabel('$g(\\tau)$ for %s'%labels[i])
        plt.tight_layout()
        plt.savefig(fname)
        print('Done!')




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

    def exponential_fit(self, ac, use_function='single'):
        """Calls on `single_exp_decay` or `double_exp_decay` for an
        exponential fitting of an autocorrelation curve.

        :param ac:
        :param default='single' use_function:
        :return np.array yFit: the y-values of the fit curve."""

        nsteps = ac.shape[0]
        if use_function == 'single':
            v0 = [0.0, 1.0 , 4000.]  # Initial guess [a0, a1, tau1] for a0 + a1*exp(-(x/tau1))
            popt, pcov = curve_fit(self.single_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
            yFit_data = self.single_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2])
            print('Best-fit tau1 = %s +/- %s'%(popt[2],pcov[2][2]))
            max_tau = popt[2]
        else:
            v0 = [0.0, 0.9, 0.1, 4000., 200.0]  # Initial guess [a0, a1,a2, tau1, tau2] for a0 + a1*exp(-(x/tau1)) + a2*exp(-(x/tau2))
            popt, pcov = curve_fit(self.double_exp_decay, np.arange(nsteps), ac, p0=v0, maxfev=10000)  # ignore last bin, which has 0 counts
            yFit_data = self.double_exp_decay(np.arange(nsteps), popt[0], popt[1], popt[2], popt[3], popt[4])
            print('Best-fit tau1 = %s +/- %s'%(popt[3],pcov[3][3]))
            print('Best-fit tau2 = %s +/- %s'%(popt[4],pcov[4][4]))
            #NOTE: This may need to be fixed later
            max_tau = max(popt[3],popt[4])
        return yFit_data,max_tau

    def cal_auto(self, data):
        """Calculates the autocorrelation"""

        print('Calculating autocorrelation ...')
        max_tau=10000
        autocorrs = []
        for timeseries in data:
            autocorrs.append( self.g(np.array(timeseries), max_tau=self.maxtau) )

        print('Done!')
        return autocorrs

    def g(self, f, max_tau = 10000, normalize=True):
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


    def autocorrelation_time(self, autocorr):
        """Computes the autocorrelation time:

        :math:`\tau_{0} = \int g(\tau) d\tau`
        """
        result = [sum(autocorr[i]) for i in range(len(autocorr))]
        return np.array(result)

    def get_blocks(self, data, nblocks=5):
        """Method used to partition data into blocks"""

        # slice the data into nblocks
        blocks = []
        for vec in data:
            dx = int(len(vec)/nblocks)
            blocks.append([vec[dx*n:dx*(n+1)] for n in range(nblocks)])
        return blocks


    def get_autocorrelation_curves(self, method="block-avg", nblocks=5,
            plot_traces=True):
        """Compute autocorrelaton function for a time-series f(t), partition the
        data into the specified number of blocks and plot the autocorrelation curve.

        :param string method: method for computing autocorrelation time; "block-avg" or "exp"
        :param int nblock: number of blocks to split up the trajectory
        :param bool default=True plot_traces: will plot the trajectory traces
        :return figure: A figure of autocorrelation curves for each restraint
        """

        sampled_parameters = self.sampled_parameters
        maxtau = self.maxtau
        # C++
        #autocorr = np.array(c_conv.autocorrelation(sampled_parameters,
        #        int(maxtau), bool(normalize)))
        #tau_c = np.array(c_conv.autocorrelation_time(autocorr))

        # Python
        autocorr = self.cal_auto(sampled_parameters)
        tau_c = self.autocorrelation_time(autocorr)

        if method == "block-avg":
            blocks = self.get_blocks(sampled_parameters, nblocks)
            x,y = [],[]
            for i in range(len(blocks)):
                y.append(self.cal_auto(blocks[i]))
                x.append(self.autocorrelation_time(y[i]))
            self.autocorr = np.average(y, axis=1)
            self.tau_c = np.average(x, axis=1)
            # Check to see if there are any negative autocorrelation times
            if any(i < 0 for i in self.tau_c):
                print("NOTE: Found a negative autocorrelation time...")
            std_y = np.std(y, axis=1)
            std_x = np.std(x, axis=1)
            self.plot_auto_curve(self.autocorr, self.tau_c, self.labels,
                    std_x=std_x, std_y=std_y)

        if method == "exp":
            self.autocorr = autocorr
            yFits,popts = [],[]
            for i in range(len(self.autocorr)):
                yFit,popt = self.exponential_fit(self.autocorr[i])
                popts.append(popt)
                yFits.append(yFit)
            self.tau_c = np.max(popts)
            self.plot_auto_curve_with_exp_fitting(self.autocorr, yFits, self.labels)

        if plot_traces:
            self.plot_traces()


    def process(self, nblock=5, nfold=10, nrounds=100, savefile=True,
            plot=True, verbose=False, block=False, normalize=True):
        #NOTE: nrounds should be more general look at self.nrounds...in the __init__ function
        """Process the trajectory by computing the autocorrelation, fitting with
        an exponential, plotting the traces, etc...

        :param int nblock: number of blocks
        :param int nfold: number of
        :param int nrounds: number of rounds to bootstrap
        :param bool default=True savefile:
        :param bool default=True plot:
        :param bool default=False block: block averaging
        :param bool verbose: verbosity
        """

        if block:
            r_total = [[] for i in range(len(self.rest_type))]
            r_max = [[] for i in range(len(self.rest_type))]
            for i in range(len(self.tau_c)):
                tau_auto = self.tau_c[i]
                tau = int(1+2*tau_auto)
                T_new = self.traj['trajectory'][::tau]
                nsnaps = len(T_new)
                dx = int(nsnaps/self.nfold)
                for subset in range(self.nblock):
                    T_total = T_new[dx*subset:dx*(subset+1)]
                    #for j in range(len(self.rest_type)):
                    r_grid = np.zeros(len(self.allowed_parameters[i]))
                    for k in T_total:
                        ind = np.concatenate(k[4])[i]
                        r_grid[ind]+=1
                    r_total[i].append(r_grid)
                    r_max[i].append(self.allowed_parameters[i][np.argmax(r_grid)])

            self.plot_block_avg(nblock,r_max)
        all_JSD=[[] for i in range(len(self.tau_c))]      # create JSD list
        all_JSDs=[[[] for i in range(self.nfold)] for j in range(len(self.tau_c))]   # create JSD list of distribution
        print('starting calculating JSDs ...')
        for i in range(len(self.tau_c)):
            ind = i
            tau_auto = self.tau_c[i]
            tau = int(1+2*tau_auto)
            T_new = self.traj['trajectory'][::tau]
            nsnaps = len(T_new)
            dx = int(nsnaps/self.nfold)
            for subset in range(self.nfold):
                half = dx * (subset+1)/2
                T1 = T_new[:half]     # first half of the trajectory
                T2 = T_new[half:dx*(subset+1)]    # second half of the trajectory
                T_total = T_new[:dx*(subset+1)]     # total trajectory
                all_JSD[i].append(self.compute_JSD(T1,T2,T_total,ind,self.allowed_parameters[i]))   # compute JSD
                for r in range(self.nround):      # now let's mix this dataset
                    mT1 = np.random.choice(len(T_total),len(T_total)/2,replace=False)    # randomly pickup snapshots (index) as the first part
                    mT2 = np.delete(np.arange(0,len(T_total),1),mT1)           # take the rest (index) as the second part
                    temp_T1, temp_T2 = [],[]
                    for snapshot in mT1:
                            temp_T1.append(T_total[snapshot])      # take the first part dataset from the trajectory
                    for snapshot in mT2:
                            temp_T2.append(T_total[snapshot])      # take the second part dataset from the trajectory
                    all_JSDs[i][subset].append(self.compute_JSD(temp_T1,temp_T2,T_total,ind,self.allowed_parameters[i]))
        if savefile:
            np.save("all_JSD.npy", all_JSD)
            np.save("all_JSDs.npy", all_JSDs)
        print('Done!')
        self.plot_JSD_distribution(np.array(all_JSD), np.array(all_JSDs), nrounds)
        self.plot_JSD_conv(np.array(all_JSD), np.array(all_JSDs))


    def plot_block_avg(self, nblock, r_max, fname = "block_avg.png"):
        plt.figure(figsize=(10,5*len(self.rest_type)))
        x=np.arange(1.,nblock+1.,1.)
        colors=['red', 'blue','black','green']
        for i in range(len(self.rest_type)):
            total_max = self.allowed_parameters[i][np.argmax(self.traj['sampled_parameters'][i])]
            plt.subplot(len(self.rest_type),1,i+1)
            plt.plot(x,r_max[i],'o-',color=colors[i],label=self.labels[i])
            plt.xlabel('block')
            plt.ylabel('allowed '+self.labels[i])
            plt.ylim(min(self.allowed_parameters[i]),max(self.allowed_parameters[i]))
            plt.plot(nblock-0.2,total_max,'*',ms=20,color='green',label='total max')
            plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fname)


    def compute_JSD(self, T1,T2,T_total,ind,allowed_parameters):
        """Compute JSD for a given part of trajectory.

        :var T1, T2, T_total: part 1, part2 and total (part1 + part2)
        :var rest_type: experimental restraint type
        :var allowed_parameters: nuisacne parameters range
        :return float: Jensen–Shannon divergence
        """

#        all_JSD = np.zeros(len(rest_type))
#        for i in range(len(rest_type)):
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

    def plot_JSD_conv(self, JSD, JSDs, p_limit=0.99):
        """Plot Jensen–Shannon divergence (JSD) distribution for convergence check.

        :var all_JSD: JSDs for different amount of total dataset
        :var all_JSDs: JSDs for different amount of total dataset from bootstrapping
        :var rest_type: experimental restraint type
        :return figure: A figure of JSD and JSDs distribution
        """

        print('plotting JSDs ...')
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

#                ax = plt.gca()
#                anchored_text = AnchoredText('%0.3f'%(p[ind]/norm),
#                        loc="upper left",frameon=False)
#                ax.add_artist(anchored_text)
                plt.ylim(bottom=0.0, top=1.0)
                plt.xlim(left=0.0)
                plt.xlabel('JSD')
                plt.ylabel('$p$')
                plt.xticks(fontsize=6)
                plt.locator_params(axis='x',nbins=5)
                plt.title('%d'%(10*(j+1))+'%',fontsize=10)
                plt.legend(loc='best',fontsize=6)
            plt.tight_layout()
            plt.savefig('JSD_conv_%s.pdf'%self.rest_type[k])
        print('Done')



    def plot_JSD_distribution(self, all_JSD, all_JSDs, nrounds, fname="JSD_distribution.png"):
        """Plots the distributions for JSD
        """
        print(all_JSDs.shape)
        print(all_JSDs[0].shape)
        print(all_JSDs[0][0].shape)

        colors=['red', 'blue','black','green']
        # convert shape of all_JSD from (fold,n_rest) to (n_rest,fold)
        n_rest = len(self.rest_type)
        # compute mean, std of JSDs from each fold dataset of each restraint
        JSD_dist = [[] for i in range(n_rest)]
        JSD_std = [[] for i in range(n_rest)]
        for rest in range(n_rest):
            for f in range(self.nfold):
                temp_JSD = []
                for r in range(self.nround):
                    temp_JSD.append(all_JSDs[rest][f][r])
                JSD_dist[rest].append(np.mean(temp_JSD))
                JSD_std[rest].append(np.std(temp_JSD))
        plt.figure(figsize=(10,5*n_rest))
        # NOTE: To Yunhui, can we generalize this next line using nfolds?
        x=np.arange(10.,101.,10.)   # the dataset was divided into ten folds (this is the only hard coded part)
        for i in range(n_rest):
            plt.subplot(n_rest,1,i+1)
            plt.plot(x,all_JSD[i].transpose(),'o-',color=colors[i],label=self.labels[i])
            plt.hold(True)
            #plt.plot(x,JSD_dist[i],'*',color=colors[i],label=self.labels[i])
            plt.fill_between(x,np.array(JSD_dist[i])+2*np.array(JSD_std[i]),
                    np.array(JSD_dist[i])-2*np.array(JSD_std[i]),
                    color=colors[i],alpha=0.2)
            plt.xlabel('dataset (%)')
            plt.ylabel('JSD')
            plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fname)



