##############################################################################
# Authors: Yunhui Ge, Rob Raddi
# This file includes functions not part of the source code but will be useful
# in different cases.
##############################################################################


##############################################################################
# Imports
##############################################################################



import sys, os, glob
import numpy as np
import re
import yaml, io
#
import mdtraj as md
import glob
#from J_coupling import * # MDTraj altered src code
import scipy ############################# Linest
from scipy import stats
import matplotlib ######################## Plotting
# Import Publication Style Fonts for Figures:
matplotlib.use('Agg')
fontfamily={'family':'sans-serif','sans-serif':['Arial']}
from matplotlib import pyplot as plt
plt.rc('font', **fontfamily)
import matplotlib.cm as cm


##############################################################################
# Code
##############################################################################

def sort_data(dataFiles):
    '''Sorts your input data regardless of the data type
    (experimental observable).
    '''

    dir_list=[]
    if not os.path.exists(dataFiles):
                raise ValueError("data directory doesn't exist")
    if ',' in dataFiles:
        print 'Sorting out the data...\n'
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
#    print 'dir_list', dir_list

    data = [[] for x in xrange(7)] # list for every extension; 7 possible experimental observables supported
    # Sorting the data by extension into lists. Various directories is not an issue...
    for i in range(0,len(dir_list)):
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        # This convert / sorted glob is a bit fishy... needs many tests
        for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
            if j.endswith('.noe'):
                data[0].append(j)
            elif j.endswith('.J'):
                data[1].append(j)
            elif j.endswith('.cs_H'):
                data[2].append(j)
            elif j.endswith('.cs_Ha'):
                data[3].append(j)
            elif j.endswith('.cs_N'):
                data[4].append(j)
            elif j.endswith('.cs_CA'):
                data[5].append(j)
            elif j.endswith('.pf'):
                data[6].append(j)
            else:
                raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
    data = np.array(filter(None, data)) # removing any empty lists
    Data = np.stack(data, axis=-1)
    data = Data.tolist()
    return data

def read_yaml(yaml_file):
    '''Opens a yaml file and parses the output nicely.'''

    with io.open(yaml_file,'r') as file:
        loaded_data = yaml.load(file)
        print('%s'%loaded_data).replace(" '","\n\n '")



def rmsd(v,w):
    '''Computes the RMSD of ref to theoretical.'''

    diffs = []
    n = len(v)
    for i in range(n):
        diffs.append( (v[i] - w[i])**2. )
    result = np.sqrt(1./n *np.sum(np.array(diffs)))
    print result
    return '%0.2f'%result


def Plot(x,y,models,xlabel='x',ylabel='y',name=None,
        color=False,fig_size=(3.3,3),
        xmin=None,xmax=None,ymin=None,ymax=None):
    '''Saves a plot with statistical information.'''

    marks = ['o','D','2','>','*',',',"4","8","s",
             "p","P","*","h","H","+","x","X","D","d"]
    colors = ['b','k','g','r','m','c','y']

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # get RMSD
    RMSD = []
    for i in range(len(y)):
        print 'RMSD for ',models[i]
        RMSD.append(rmsd(v=y[i],w=x))

    # Get statistics
    J = 0
    R = []
    for i in range(len(x)):
        z = np.polyfit(x[i], y[i], deg=1)
        n_coeff = len(z)
        p = np.poly1d(z)
        print 'Model: ',models[i]
        print 'LINEST data:'
        print "y=%.6f*x+(%.6f)"%(z[0],z[1])
        print scipy.stats.linregress(x[i],y[i])
        print scipy.stats.chi2(y[i])
        print scipy.stats.ttest_ind(x[i], y[i], axis=0, equal_var=True)
        print 'R^2 = ',(scipy.stats.linregress(x[i],y[i])[2])**2.
        #r2 = r'$R^{2}$ = %.4f'%((scipy.stats.linregress(x[i],y[i])[2])**2.)
        R.append('%0.2f'%scipy.stats.linregress(x[i],y[i])[2])
        # Uncomment below to plot the trendlines:
        #ax.plot(x[i],p(x[i]),color=colors[J])
#NOTE: We need to add error bars to the trendlines
        J += 1

    for i in range(len(x)):
        for k in range(len(x[i])):
            if k == len(x[i])-1:
                ax.scatter(x[i][k],y[i][k],color=colors[i],label='%s, RMSE = %s, R = %s'%(models[i],RMSD[i],R[i]),s=2)
            else:
                ax.scatter(x[i][k],y[i][k],color=colors[i],s=2)

    # Obtain a basic reference line
    ax.plot([np.min(x),np.max(x)],[np.min(x),np.max(x)],color='gold')

    # Set the x and y axis limits:
    ax.set_xlim(left=xmin,right=xmax)
    ax.set_ylim(top=ymax,bottom=ymin)

    # Edit the plot features
    ax.set_xlabel('%s'%xlabel, fontsize=8)
    ax.set_ylabel('%s'%ylabel, fontsize=8)
    leg1 = ax.legend(loc='best', fontsize=5,fancybox=True, framealpha=0,markerscale=0)
    J = 0
    for text in leg1.get_texts():
        text.set_color(colors[J])
        J += 1

    # Setting the ticks and tick marks
    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(8)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=8)
            #mark.set_rotation(s=15)
    fig.tight_layout()
    if name==None:
        pass
    else:
        fig.savefig('%s'%name)
    fig.show()





