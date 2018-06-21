import sys, os, glob
from toolbox import *
models=["Ruterjans1999","Bax2007","Bax1997","Habeck" ,"Vuister","Pardi"]
gro=['8690.gro','8693.gro','8696.gro','8699.gro']
for i in range(1,5):
    print 'ligand', i
    os.chdir('ligand%d_xtc'%i)
    for j in models:
        print 'model', j
        os.system('mkdir %s'%j)
        files=glob.glob('*xtc')
        list1=sorted(files,key=lambda x: int(os.path.splitext(x.split("traj")[1])[0]))
        for k in list1:
            print k
            a=get_J3_HN_HA('%s'%k,gro[i-1],model='%s'%j, outname = '%s/%s'%(j,k.split('.')[0]))
    os.chdir('../')
        
