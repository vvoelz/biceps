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
#from J_coupling import * # MDTraj altered src code
from KarplusRelation import *
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
##############################################################################
# Code
##############################################################################

def sort_data(dataFiles):
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
    data = np.array(filter(None, data)) # removing any empty lists
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
                scheme.append('gamma')
            else:
                raise ValueError("Incompatible File extension. Use:{*.noe, *.J, *.cs_H, *.cs_Ha, *.cs_N, *.cs_Ca, *.pf}")

    return scheme

def write_results(self, outfilename):
    """Writes a compact file of several arrays into binary format."""

    np.savez_compressed(outfilename, self.results)

def read_results(filename):
    """Reads a npz file"""

    loaded = np.load(filename)
    print loaded.items()

def convert_pop_to_energy(pop_filename, out_filename=None):
    """Convert population to energy for each state using U = -np.log(P)"""
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
    Parameters
    ----------
    traj: trajectory file
    top: topology file
    frame: specific frame for computing
    model: Karplus coefficient models ["Ruterjans1999","Bax2007","Bax1997","Habeck" ,"Vuister","Pardi"]
    outname: if not None, the output will be saved and a file name (in the format of string) is required.
    '''
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
        J = compute_JS_HN_HA(t, model = model)
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
    """Calculate the signed dihedral angle between 4 positions.  Result is in degrees."""
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
    output = 'traj_lambda0.00.npz'
    if debug:
            print 'Loading %s ...'%output
    results = np.load( file(output, 'r') )['arr_0'].item() 
    n_restraints = len(results['ref_potential'])
    for i in range(n_restraints):
        if results['ref_potential'][i][0] == 'Nan':
            pass
        else:
            n_model = len(results['ref_potential'][i][0])
            c,r = 5, int(n_model)/5 + 1
            x = np.arange(0.0,30.0,0.01)
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



def get_rest_type(traj):
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
            

def autocorr_valid(x,tau):
    t = tau 
    y = x[:np.size(x)-t]
    g = np.correlate(x, y, mode='valid')
    n = np.array([np.size(x)-t]*len(g))
    return g/n


def compute_ac(traj,tau,rest_type=None,allowed_parameters=None):
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
    '''Compute JSD for a given part of trajectory.
    Parameters
    ----------
    T1, T2, T_total: part 1, part2 and total (part1 + part2)
    traj: trajectory from BICePs sampling
    '''
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

