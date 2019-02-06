import sys, os
import numpy as np
from matplotlib import pyplot as plt
l=np.arange(0.0,6.0,0.5)

#l=np.arange(0.0,10.5,0.5)
#path=["1M_1", "1M_2", "1M_3"]
path_E=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12']
path=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12','RUN13','RUN14','RUN15','RUN16','RUN17','RUN18']
path_G=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12','RUN13','RUN14','RUN15','RUN16','RUN17','RUN18','RUN19','RUN20','RUN21','RUN22','RUN23']
#path_G=['RUN16','RUN17','RUN18','RUN19','RUN20','RUN21','RUN22','RUN23']
#path=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6']
if (1):
	g=dict()	#onlyE
	f=dict()	#onlyG
	h=dict()	#no_ref
	error_E=[]
	point_E=[]
        error_G=[]
        point_G=[]
        error=[]
        point=[]

	for i in range(len(l)):
		g[i]=[]
		f[i]=[]
		h[i]=[]
		for j in range(len(path)):
			c=np.loadtxt('micro_noref/%s/BF_ref_normal.dat'%(path[j]))
                        h[i].append(c[i][0])
                point.append(np.mean(h[i]))
                error.append(np.std(h[i]))

		for k in range(len(path_E)):
                        a=np.loadtxt('micro_onlyE/%s/BF_ref_normal.dat'%(path_E[k]))
                        g[i].append(a[i][0])
                point_E.append(np.mean(g[i]))
                error_E.append(np.std(g[i]))

		for m in range(len(path_G)):
                        b=np.loadtxt('micro_onlyG/%s/BF_ref_normal.dat'%(path_G[m]))
			f[i].append(b[i][0])
                point_G.append(np.mean(f[i]))
                error_G.append(np.std(f[i]))

	plt.figure(figsize=(12,8))
	plt.xticks(l)
	plt.xlim(0,7.0)
#	plt.plot(l,point,"-o")

	plt.errorbar(l,point,yerr=error,fmt="-o",color='red',label='no_ref')
	plt.errorbar(l,point_E,yerr=error_E,fmt="-o",color='blue',label='exp')
	plt.errorbar(l,point_G,yerr=error_G,fmt="-o",color='green',label='Gau')
	plt.xlabel(r'$\epsilon$ values')
	plt.ylabel('BICePs scores')
	plt.title(r'BICePs scores change along with different $\epsilon$ values')
	plt.legend(loc="upper right")
	plt.savefig('micro_whole.png')
	plt.show()
	sys.exit()




BS=[]
for i in l:
	b=np.loadtxt('results/micro/%.1f/BF_ref_normal.dat'%i)
	BS.append(b[1][0])
#print len(BS)
#print BS
plt.figure(figsize=(12,8))
plt.xticks(l)
plt.plot(l,BS,"-o")
plt.xlabel(r'$\epsilon$ values')
plt.ylabel('BICePs scores')
plt.title(r'BICePs scores change along with different $\epsilon$ values')
#plt.savefig('BS_gaussian_diffglobal_micro.pdf')
plt.show()






	
	
