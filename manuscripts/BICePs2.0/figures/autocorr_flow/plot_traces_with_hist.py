# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
#import c_convergence as c_conv

def get_sampled_parameters(traj):
    """Get sampled parameters along time (steps).

    :return list: A list of all nuisance paramters sampled
    """

    parameters = []
    rest_type = traj['rest_type']
    for i in range(len(rest_type)):
        parameters.append(np.array(traj['traces'])[:,i])
    parameters = np.array(parameters)
    return parameters



def plot_traces(traj, fname="traj_traces.png", xlim=None):

    print('Plotting traces...')
    rest_type = traj['rest_type']
    n_rest = len(rest_type)
    allowed_parameters = traj['allowed_parameters']
    sampled_parameters = get_sampled_parameters(traj)
    labels = ["$\\sigma$","$\\gamma$"]
    fig = plt.figure(figsize=(9,4.5))
    for i in range(1,2):#len(rest_type)):
        total_steps = len(sampled_parameters[i])
        x = np.arange(1,total_steps+0.1,1)
        grid = plt.GridSpec(3,3, hspace=0.01, wspace=0.01)
        #main_ax = fig.add_subplot()#grid[1:,1:])
        y_hist = fig.add_subplot(grid[1, 2:])
        y_hist.hist(sampled_parameters[i],
                #histtype='stepfilled',
                #histtype='barstacked',
                histtype='bar',
                #histtype='stop',
                orientation='horizontal', color='blue',
                edgecolor='black', linewidth=1.2)
        #y_hist.invert_xaxis()
        y_hist.yaxis.set_label_position('right')
        y_hist.yaxis.set_ticks_position('right')
        y_hist.xaxis.set_label_position('top')
        y_hist.xaxis.set_ticks_position('top')
        y_hist.locator_params(axis='y', nbins=3)
        y_hist.locator_params(axis='x', nbins=3)
        y_hist.set_xlabel('counts', fontsize=14)
        #y_hist.set_ylim(left=np.min(allowed_parameters[0]),right=np.max(allowed_parameters[0]))

        x_trace = fig.add_subplot(grid[1, 0:2], sharey=y_hist)
        x_trace.plot(x, sampled_parameters[i],label=labels[i])
        x_trace.xaxis.set_label_position('bottom')
        x_trace.yaxis.set_label_position('left')
        x_trace.yaxis.set_ticks_position('left')
        x_trace.set_ylabel(labels[i], fontsize=14)
        x_trace.set_xlabel('steps', fontsize=14)
        if xlim:
            x_trace.set_xlim(left=min(x), right=max(x))

    plt.tight_layout()
    plt.savefig(fname)
    print('Done!')


if __name__ == "__main__":

    testing = False
    if testing:
        trajfile = "/Volumes/WD_Passport_1TB/new_sampling/new_sampling/d_1.01/results_ref_normal_10000/traj_lambda1.00.npz"
    else:
        trajfile = "traj_lambda1.00.npz"
    traj = np.load(trajfile)['arr_0'].item()
    plot_traces(traj, fname="traj_traces_with_hist.pdf", xlim=[0,1000000])

