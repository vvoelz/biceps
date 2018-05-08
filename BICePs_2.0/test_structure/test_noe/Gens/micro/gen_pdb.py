import mdtraj as md
a=md.load('microstates.pdb')
for i in range(15037):
	print i
	a[i].save_pdb('%d.pdb'%i)
