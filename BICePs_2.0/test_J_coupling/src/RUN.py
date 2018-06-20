
# Libraries:{{{
import numpy as np
import mdtraj as md
import glob
from J_coupling import * # MDTraj altered src code
import scipy ############################# Linest
from scipy import stats
import matplotlib ######################## Plotting
# Import Publication Style Fonts for Figures:
matplotlib.use('Agg')
fontfamily={'family':'sans-serif','sans-serif':['Arial']}
from matplotlib import pyplot as plt
plt.rc('font', **fontfamily)
import matplotlib.cm as cm
#}}}

n = 4 # Ligand Number
N = n-1 # Index for Ligand
get_data = True
fname = 'ligand%s'%n # Filename for outputs
gfiles = [8690,8693,8696,8699] # list of grofile names

# List of all the Karplus coefficient Models
Models = ["Ruterjans1999","Bax2007","Bax1997","Habeck" ,"Vuister","Pardi"]

# Set paths
data = '/Users/tuc41004/Desktop/nmr-biceps/BICePs_2.0/test_J_coupling/%s/'%fname
gro = data+'%s.gro'%gfiles[N]
top = md.load_topology(gro)
#trajs = [data + 'traj%s.xtc'%i for i in range(len(glob.glob(data+'traj*.xtc')))]
trajs = sorted(glob.glob(data+'traj*.xtc'))

# Experimental Data from Erdelyi et al - Table S5.
exp_1 = np.array([7.9,7.3,0,7.7,8.4,0,0,0,0])
exp_2 = np.array([0,7.4,0,6.2,7.4,0,7.8,0,0])
exp_3 = np.array([7.3,8.6,0,8.0,8.2,7.3,7.5,0,0])
exp_4 = np.array([6.6,7.3,0,6.8,0,7.4,0,0,0])


# Get Theoretical:{{{
if get_data == True:
    J = {}       # Dictionary of all J3_HN_HA values for each model
    J_val = {}   #
    J_3_exp = [] #
    results = [] #
    # Compute the Values for J coupling const.
    for j in range(len(Models)):
        for i in range(len(trajs)):
            t = md.load(trajs[i],top=top)[0]
            name = trajs[i].split('/')[-1].split('.')[0]
            print 'Computing 3J_Ha_HN with model %s for %s ...'%(Models[j],trajs[i].split('/')[-1])
            J["%s"%Models[j]+'_'+name] = compute_J3_HN_HA(t, model=Models[j])

        # Print out the residues names involved:
        ind = J["%s"%Models[j]+'_'+name][0] # number of residues?
        for i in range(len(ind)):
            for m in range(len(ind[i])):
                atom = t.top.atom(index=ind[i][m])
                if m == len(ind[i])-1:
                    J_val["%s"%(str(atom).split('-')[0])] = J["%s"%Models[j]+'_'+name][1][0][i]

        # Computing the average values of all trajectories for each residue:
        model_result = []
        for i in range(len(ind)):
            for n in range(len(ind[i])):
                atom = t.top.atom(index=ind[i][n])
                if n == len(ind[i])-1:
                    avg = np.average(J_val["%s"%(str(atom).split('-')[0])])
                    print 'Model %s : Avg J : %s = '%(Models[j],str(atom).split('-')[0]), avg
                    model_result.append(avg)
        results.append(model_result)

    np.save('%s.npy'%fname,results)
# }}}

# Methods:{{{

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
    '''Returns a plot '''

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

#}}}

# Get Plot:{{{
# Set the x and y data:
x = np.array([exp_1,exp_2,exp_3,exp_4]) # All Experimental Data
x = x[N] # Using only the experimental data for the specified Ligand
results = np.load(fname+'.npy')
y = results #np.array(results)

# Remove the data that correspond to experimental data equal to zero
X,Y = [],[]
for i in range(len(x)):
    if x[i] != 0:
        X_,Y_ = [],[]
        for k in range(len(y)):
            X_.append(float(x[i]))
            Y_.append(float(y[k][i]))
        X.append(X_)
        Y.append(Y_)
x,y = np.array(X),np.array(Y)
x,y = np.transpose(x),np.transpose(y)

# Plot the results for the specified ligand:
Plot(x=x,y=y,xlabel='Experiment',
        ylabel='Prediction',name=fname+'.pdf',
        models=Models,xmin=6.0,xmax=9.0,ymin=4.5,ymax=10.5)

#}}}






