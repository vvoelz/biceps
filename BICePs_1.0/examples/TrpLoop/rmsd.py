import mdtraj as md
t = md.load('TrpLoop2b_nmr/Gens/Gens509.pdb')
#t2=md.load('3.pdb')
Ind = t.topology.select("backbone or name == 'CB'")
#Ind2 = t2.topology.select("backbone or name == 'CB'")
#print len(Ind), len(Ind2)
t1 = t.atom_slice(Ind)
#t3=t2.atom_slice(Ind2)
t4=md.load('b_4.pdb')
r=md.rmsd(t4,t1)
print r
