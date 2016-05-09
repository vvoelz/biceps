import mdtraj as md
import numpy as np
import pandas as pd
pd.options.display.max_rows=180
traj = md.load("traj0.xtc", top="conf.gro")
shifts = md.nmr.chemical_shifts_shiftx2(traj, pH = 2.5, temperature = 280.0)
pd.DataFrame.to_pickle(shifts,"shifts_total.txt")
print shifts.mean(axis=1)
