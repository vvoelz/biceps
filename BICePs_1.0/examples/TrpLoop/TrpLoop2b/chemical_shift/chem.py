import mdtraj as md
import numpy as np
for i in range(1000):
	traj = md.load("../Gens/Gens%d.pdb"%i)
	shifts = md.nmr.chemical_shifts_shiftx2(traj, pH=2.5, temperature = 280.0)
	np.savetxt("cs_state%d.txt"%i,shifts.mean(axis=1))
