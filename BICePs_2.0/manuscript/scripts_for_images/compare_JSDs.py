import numpy as np
import glob,os,datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# User specified paths
base_dir = "/Volumes/RMR_4TB/new_sampling/new_sampling/"
numpyFile = "all_JSD.npy"
full_path = base_dir+"d_*/results_ref_normal_*/"+numpyFile
n_restraints = 2

def condition1(path,f1,f2):
    """Check to see if a file in the directory exists by replacing
    f1 with f2 for a given path."""
    return os.path.exists(path.replace(f1,f2))

def condition2(path,f1,f2):
    """Check to see if the timestamps match from last metadata change."""
    ans = 0
    t1 = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    t2 = datetime.datetime.fromtimestamp(os.path.getmtime(path.replace(f1,f2)))
    date1 = str(t1).split()[0]
    date2 = str(t1).split()[0]
    if date1 == date2:
        ans = 1
    return ans


def plot_JSD_comparison(steps, JSD, restraints,
        fig_name="JSD_comparison.png"):
    """plot that compares JSDs of 100% data distribution against number of steps"""

    n_restraints = len(restraints)
    plt.figure(figsize=(10,5*n_restraints))
    for i in range(n_restraints):
        ax = plt.subplot(n_restraints,1,i+1)
        plt.plot(steps[i], JSD[i],'o-',color='k')
        plt.xscale("log")
        plt.xlabel('Steps', fontsize=18)
        plt.ylabel('%s for %s'%('JSD', restraints[i]), fontsize=18)
        ticks = [ax.xaxis.get_minor_ticks(),
                 ax.xaxis.get_major_ticks()]
        marks = [ax.get_xticklabels(),
                ax.get_yticklabels()]
        for k in range(0,len(ticks)):
            for tick in ticks[k]:
                tick.label.set_fontsize(18)
        for k in range(0,len(marks)):
            for mark in marks[k]:
                mark.set_size(fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_name)



# Get all the directories that satisfy the conditions
files,FILES = glob.glob(full_path),[]
count = 0
for i in range(len(files)):
    file = files[i]
    if condition1(file, f1=numpyFile, f2="JSD_distribution.png"):
        if condition2(file, f1=numpyFile, f2="JSD_distribution.png"):
            FILES.append(file)
            count += 1
print("%s of %s"%(count,len(files)))
del(files)


# Create a bunch of sublists that sort the directories by the different number of steps
FILES_sublists = []
subs = []
for i in range(len(FILES)):
    file = FILES[i]
    subdir = file.split("results_ref_normal")[0]
    if len(FILES_sublists) >= 1:
        if str(subdir) in FILES_sublists[-1][-1]:
            subdir = file.split("results_ref_normal")[0]
            FILES_sublists[-1].append(file)
        else:
            FILES_sublists.append([file])
            subs.append(subdir.split("/")[-2])
    else:
        FILES_sublists.append([file])
        subs.append(subdir.split("/")[-2])


# Now put the data into sublists that sorts the dirs
index = -1 # Index of the data point from the file
x,y = [],[]
for i in range(len(FILES_sublists)):
    x.append([])
    y.append([])
    for j in range(len(FILES_sublists[i])):
        file = FILES_sublists[i][j]
        x[-1].append([int(file.split("/")[-2].split("_")[-1])for k in range(n_restraints)])
        y[-1].append([float(np.load(file)[k][-1]) for k in range(n_restraints)])

    X = np.array(x[i]).transpose()
    Y = np.array(y[i]).transpose()
    restraints = ["$\sigma$","$\gamma$"]
    plot_JSD_comparison(X, JSD=Y, restraints=restraints,
            fig_name="JSD_comparison_%s.pdf"%(subs[i].replace(".","-")))








