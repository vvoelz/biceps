import sys, os
sys.path.append('biceps')
from toolbox import *

#traj = np.load('results_ref_normal/traj_lambda1.00.npz')['arr_0'].item()
#grid = traj['grid']

#for i in range(len(grid)):
#    plt.figure()
#    cmap=plt.get_cmap('Blues')
#    plt.pcolor(grid[i],cmap=cmap,vmin=0.0,vmax=np.max(grid[i]),edgecolors='none')
#    plt.colorbar()
#    plt.savefig('test%d.pdf'%i)
#    plt.close()

g = plot_grid('results_ref_normal/traj_lambda1.00.npz')


