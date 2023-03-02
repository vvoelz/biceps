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


def plot_JSD_distribution(all_JSD, all_JSDs, nround, nfold, fname="JSD_distribution.png"):
    """Plots the distributions for JSD"""

    rest_type = ['sigma_noe', 'gamma']
    colors=['red', 'blue','black','green']
    # convert shape of all_JSD from (fold,n_rest) to (n_rest,fold)
    n_rest = len(rest_type)
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
    #plt.figure(figsize=(10,5*n_rest))
    plt.figure( figsize=(3*len(rest_type),6))
    x=np.arange(int(100/nfold),101.,int(100/nfold))   # the dataset was divided into ten folds (this is the only hard coded part)
    for i in range(n_rest):
        plt.subplot(n_rest,1,i+1)
        plt.plot(x,all_JSD[i].transpose(),'o-',color=colors[i],label=["$\\sigma$","$\\gamma$"][i])
        plt.hold(True)
        # 2 Standard deviations from the mean
        plt.fill_between(x,np.array(JSD_dist[i])+2*np.array(JSD_std[i]),
                np.array(JSD_dist[i])-2*np.array(JSD_std[i]),
                color=colors[i],alpha=0.2)

        # at 95% confidence interval
        bounds = np.sort(all_JSDs[i])
        # remove top 50 and lower 50
        lower = bounds[:, int(nround*0.05)]
        upper = bounds[:, int(nround*0.95)]
        #plt.fill_between(x,lower,upper,color=colors[i],alpha=0.2)
        plt.xlabel('dataset (%)', size=18)
        plt.ylabel('JSD', size=18)
        plt.legend(loc='best')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(left=10, right=100)
    plt.tight_layout()
    plt.savefig(fname)



if __name__ == "__main__":

    path = "./"
    all_JSD = np.load(path+"all_JSD.npy")
    all_JSDs = np.load(path+"all_JSDs.npy")
    plot_JSD_distribution(np.array(all_JSD), np.array(all_JSDs),
            nfold=10, nround=100, fname="JSD_distribution.pdf")





