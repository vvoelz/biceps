import sys, os
import numpy as np
from matplotlib import pyplot as plt
l=np.arange(0.0,6.0,0.5)
color=['black','red','green','yellow','cyan','blue','magenta','orange','indigo','pink','darkred','indianred','springgreen','purple','goldenrod','peru','y','lightskyblue','blueviolet','gold']
#l=np.arange(0.0,10.5,0.5)
#path=["1M_1", "1M_2", "1M_3"]
path_E=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12','RUN13','RUN14','RUN15','RUN16','RUN17','RUN18','RUN19']
#path=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12','RUN13','RUN14','RUN15','RUN16','RUN17','RUN18']
#path_G=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6','RUN7','RUN8','RUN9','RUN10','RUN11','RUN12','RUN13','RUN14','RUN15','RUN16','RUN17','RUN18','RUN19','RUN20','RUN21','RUN22','RUN23']
#path_G=['RUN16','RUN17','RUN18','RUN19','RUN20','RUN21','RUN22','RUN23']
#path=['RUN1','RUN2','RUN3','RUN4','RUN5','RUN6']

if (1):
	g=dict()	#sep
	f=dict()	#whole
#	error_E=[]
#	point_E=[]
#        error_G=[]
#        point_G=[]
        error=[]
        point=[]

	for i in range(len(l)):
#		g[i]=[]
		f[i]=[]
#		h[i]=[]
		for j in range(len(path_E)):
			c=np.loadtxt('%s/BF_ref_normal.dat'%(path_E[j]))
                        f[i].append(c[i][0])
                point.append(np.mean(f[i]))
                error.append(np.std(f[i]))

	for k in range(len(path_E)):
#		g[k]=[]	
        	a=np.loadtxt('%s/BF_ref_normal.dat'%(path_E[k]))
                g[k]=list(a[:,0])
#                point_E.append(np.mean(g[i]))
#                error_E.append(np.std(g[i]))

#		for m in range(len(path_G)):
#                        b=np.loadtxt('micro_onlyG/%s/BF_ref_normal.dat'%(path_G[m]))
#			f[i].append(b[i][0])
#                point_G.append(np.mean(f[i]))
#                error_G.append(np.std(f[i]))

	plt.figure(figsize=(12,8))
	plt.xticks(l)
	plt.xlim(0,7.0)
#	plt.plot(l,point,"-o")

#	whole=plt.errorbar(l,point,yerr=error,fmt="-o",color='red',lw=3.5,ls='-.',label='whole')
#	whole[-1][0].set_linestyle('--')
	y1=[]
	y2=[]
	for i in range(len(point)):
		y1.append(point[i]-error[i])
		y2.append(point[i]+error[i])
	plt.plot(l,point,'--',color='red',linewidth=8.5,label='whole')
	plt.fill_between(l,y1,y2,alpha=0.2)
#	plt.errorbar(l,a[i],yerr=error_E,fmt="-o",color='blue',label='exp')
#	plt.errorbar(l,point_G,yerr=error_G,fmt="-o",color='green',label='Gau')
	plt.xlabel(r'$\epsilon$ values')
	plt.ylabel('BICePs scores')
	plt.title(r'BICePs scores change along with different $\epsilon$ values')
	for i in range(len(path_E)):
		        plt.errorbar(l,g[i],fmt="-o",color=color[i],label='%s'%path_E[i])
	plt.legend(loc="upper right")
	plt.savefig('test_noise.png')
#	plt.show()
	sys.exit()

