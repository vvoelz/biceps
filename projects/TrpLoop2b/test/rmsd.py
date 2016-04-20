import os, sys
import numpy as np
from matplotlib import pyplot as plt
import mdtraj as md
T = 300.0

#indices=range(121)  # only select the first 6 residues
#t = md.load('traj0.xtc',top='conf.gro')
t=md.load('Gens/Gens236.pdb')
#t = md.load_frame('traj0.xtc',199999, top ='conf.gro')
#t.save_pdb('199981.pdb')
#sys.exit(1)
Ind = t.topology.select("backbone or name == 'CB'")
#Ind = t.topology.select("(resid 5 to 10) and (backbone or name == 'CB')")
print Ind
t1 = t.atom_slice(Ind)
#t1.save_pdb('199981.pdb')
#print 't1', t1
#print 't1.xyz.shape', t1.xyz.shape
#t1.save_pdb('t1.pdb')
#print t[0]
#sys.exit(1)
#Ind2= t.topology.select("(resid 5 to 10) and (backbone or name == 'CB')")
#t4=t.atom_slice(Ind2)
#t2 = md.load("2_vav.pdb")
#Ind2 = t2.topology.select("(protein and backbone) or name == 'CB'")
#t3 = t2.atom_slice(Ind2)
#print 't3', t3
#print 't3.xyz.shape', t3.xyz.shape
#t3.save_pdb('b.pdb')
#sys.exit(1)
t3=md.load('b_4.pdb')
#t5=md.load('turn.pdb')
#len(t)
r=md.rmsd(t1,t3)
#r1=md.rmsd(t4,t5)
print r
#np.savetxt('rmsd_23.txt',r)
#sys.exit(1)
sys.exit()
print '#frame\tRMSD'
for i in np.arange(0,t1.xyz.shape[0],1):
    print i, r[i]
#sys.exit(1)

#time = np.arange(0,0.01*200001,0.01)
#print time
#try:
#plt.figure()
#plt.plot(time, r)
#plt.xlabel('time (ns)')
#plt.ylabel('rmsd (nm)')
#plt.show()
#except:
   # pass

#sys.exit(1)
np.histogram(r, bins=np.arange(0,1.01,0.01) )
counts,bins = np.histogram(r, bins=np.arange(0,1.01,0.01) )
np.histogram(r1,bins=np.arange(0,1.01,0.01))
counts1,bins=np.histogram(r1,bins=np.arange(0,1.01,0.01))
x=np.arange(0,1.01,0.01)
plt.figure(figsize=(8,8))
bincenters = (bins[0:-1]+bins[1:])/2.0
axes=plt.gca()
axes.axes.get_yaxis().set_ticks([])
plt.ylim([0,80000])
plt.plot(bincenters, counts,linewidth=3,label='All-residue')
plt.plot(bincenters,counts1,color='red',linewidth=3,label='Turn part residues')
#plt.title('RMSD Histogram of TrpLoop2a')
plt.xlabel('RMSD (nm)',fontsize=24,fontweight='bold')
plt.xticks(fontsize=24)
plt.legend(loc='upper right',fontsize=24)
plt.savefig("RMSD_96_4.pdf")
#print np.bincount(x)
#sys.exit()
plt.show()
sys.exit()
kT = 0.001987*T
f = -kT*np.log(counts)
f -= f.min()
plt.figure(figsize=(8,8))
plt.plot(bincenters, f)
plt.xlabel('rmsd (nm)')
plt.ylabel('free energy (kcal/mol)')
plt.savefig("FE_300K.pdf")
#plt.show()
