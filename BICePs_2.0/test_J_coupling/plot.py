import sys, os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
with_err=False

def rms(exp,sim):
    diff=[]
    for i in range(len(sim)):
        diff.append((exp[i]-sim[i])**2.0)
    r = np.sqrt(sum(diff)/len(sim))
    return '%.2f'%r

exp = [[8.4,7.7,7.3,7.9],[7.8,7.4,6.2,7.4],[7.5,7.3,8.2,8.0,8.6,7.3],[7.4,6.8,7.3,6.6]]
ind = [[4,3,1,0], [6,4,3,1], [6,5,4,3,1,0],[5,3,1,0]]
models=["Ruterjans1999","Bax2007","Bax1997","Habeck" ,"Vuister","Pardi"]
c=['red','green','magenta','blue','cyan','gold']
#gro=['8690.gro','8693.gro','8696.gro','8699.gro']
for i in range(1,5):
    sim=[]
    err=[]
#    R=[]
#    rmsd=[]
    print 'ligand', i
    os.chdir('ligand%d_xtc'%i)
    for j in models:
        sim.append(np.load('avg/%s_avg.npy'%j))
        err.append(np.load('avg/%s_err.npy'%j))
#    print sim
#    print err
#    sys.exit()
    real_sim=[[] for x in range(len(models))]
    real_err=[[] for x in range(len(models))]
    for k in range(len(models)):
        for l in ind[i-1]:
            real_sim[k].append(sim[k][l])
            real_err[k].append(err[k][l])
#    print real_sim
#    print real_err
#    sys.exit()
#    os.chdir('../')
    plt.figure(figsize=(3.3,3))
    for m in range(len(models)):
#        R = pearsonr(exp[i-1],real_sim[m])[0]
        R=stats.linregress(exp[i-1],real_sim[m])[2]
#        sys.exit()
#        R.append('%.2f'%(pearsonr(exp[i-1],real_sim[m])[0]))
        rmsd=(rms(exp[i-1],real_sim[m]))
        if with_err:
            plt.errorbar(exp[i-1],real_sim[m],yerr=real_err[m],fmt='o',ms=2,color=c[m],label='%s, R = %.2f, rms = %s'%(models[m], R, rmsd))
        else:
            plt.scatter(exp[i-1],real_sim[m],color=c[m],s=2,label='%s, R = %.2f, rms = %s'%(models[m], R, rmsd))
    plt.plot([6,9],[6,9],color='black',linestyle='-')
    plt.xlim(4,10)
    plt.ylim(4,10)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('exp',fontsize=6)
    plt.ylabel('sim',fontsize=6)
    leg1=plt.legend(loc='best',fontsize=4)
    n=0
    for text in leg1.get_texts():
        text.set_color(c[n])
        n+=1
    plt.savefig('../ligand%d.pdf'%i)
    plt.close()
    os.chdir('../')
