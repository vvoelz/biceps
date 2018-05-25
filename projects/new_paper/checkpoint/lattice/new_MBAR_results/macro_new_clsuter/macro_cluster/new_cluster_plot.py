import sys, os
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
l=np.arange(0.0,6.0,0.5)
#l=np.arange(0.0,10.5,0.5)
#path=["1M_1", "1M_2", "1M_3"]
color=['blue','green','red','cyan','black']
#state=['200','500','1000','5000']
state=['500','1000','5000','10000']
path=['1','2','3','4','5','6','7','8','9']
error=dict()
point=dict()
if (1):
	for k in range(len(state)):
		g=dict()
		error[k]=[]
		point[k]=[]
		for i in range(len(l)):
			g[i]=[]
			for j in range(len(path)):
				a=np.loadtxt('%s/%s/BF_ref_normal.dat'%(path[j],state[k]))
				g[i].append(a[i+1][0])
			point[k].append(np.mean(g[i]))	
			error[k].append(np.std(g[i]))

# micro 15037 states
path_G=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12','RUN13','RUN14','RUN15','RUN16','RUN17','RUN18','RUN19','RUN20','RUN21','RUN22','RUN23']
if (1):
        f=dict()        #onlyG
        error_G=[]
        point_G=[]

        for i in range(len(l)):
                f[i]=[]
                for m in range(len(path_G)):
                        b=np.loadtxt('../../micro_onlyG/%s/BF_ref_normal.dat'%(path_G[m]))
                        f[i].append(b[i][0])
                point_G.append(np.mean(f[i]))
                error_G.append(np.std(f[i]))





	plt.figure(figsize=(12,8))
	plt.xticks(l)
	plt.xlim(0,7.0)
#	plt.ylim(-10,2)
#	plt.plot(l,point,"-o")
	for k in range(len(state)):
		plt.errorbar(l,point[k],yerr=error[k],fmt="-o",color=color[k],label='%s_states'%state[k])
	plt.errorbar(l,point_G,yerr=error_G,fmt="-o",color='magenta',label='15037_states')
	plt.legend(loc='upper right')
	plt.xlabel(r'$\epsilon$ values')
	plt.ylabel('BICePs scores')
	plt.title(r'BICePs scores change along with different $\epsilon$ values')
#	plt.savefig('new_cluster_whole_no200.png')
#	plt.savefig('new_cluster.png')
	plt.show()
