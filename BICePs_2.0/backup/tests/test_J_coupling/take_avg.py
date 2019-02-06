import sys, os, glob
from toolbox import *
models=["Ruterjans1999","Bax2007","Bax1997","Habeck" ,"Vuister","Pardi"]
#gro=['8690.gro','8693.gro','8696.gro','8699.gro']
for i in range(1,5):
    print 'ligand', i
    os.chdir('ligand%d_xtc'%i)
    os.system('mkdir avg')
    for j in models:
        avg=[]
        err=[]
        total=[]
        print 'model', j
#        os.system('mkdir %s'%j)
        os.chdir('%s'%j)
        files=glob.glob('*npy')
        list1=sorted(files,key=lambda x: int(os.path.splitext(x.split("traj")[1])[0]))
        for k in list1:
            print k
            d=np.load('%s'%k)
            for y in range(len(d[1])):
                total.append(d[1][y])
        total=np.array(total)
        for z in range(9):
            avg.append(np.mean(total[:,z]))
            err.append(np.std(total[:,z]))
        np.save('../avg/%s_avg.npy'%j,avg)
        np.save('../avg/%s_err.npy'%j,err)
        os.chdir('../')
#            a=get_J3_HN_HA('%s'%k,gro[i-1],outname = '%s/%s'%(j,k.split('.')[0]))
    os.chdir('../')
        
