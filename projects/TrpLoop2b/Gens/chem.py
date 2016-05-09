import mdtraj as md
import numpy as np
import pandas as pd
pd.options.display.max_rows=999
#traj = md.load("Gens1.pdb")
#shifts = md.nmr.chemical_shifts_shiftx2(traj, pH = 2.5, temperature = 280.0)
#print shifts.mean(axis=1)
#np.savetxt("cs_state.txt",shifts.mean(axis=1))
for i in range(250):
       traj = md.load("Gens%d.pdb"%i)
       shifts = md.nmr.chemical_shifts_shiftx2(traj, pH = 2.5, temperature = 280.0)
#print shifts.mean(axis=1)
       np.savetxt("../chemical_shift/cs_state%d.txt"%i,shifts.mean(axis=1))
#np.savetxt("cs_state.txt",shifts.mean(axis=1))
