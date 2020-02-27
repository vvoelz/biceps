
# Python Libraries:{{{
import numpy as np
# Plotting modules
from matplotlib import pyplot as plt
#:}}}



def quick_plot(x, y, xlabel='x', ylabel='y',
        name=None, Type='scatter', fig_size=(12,10)):

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    if Type=='scatter':
        ax.scatter(x,y,color='k')

    if Type=='line':
        ax.plot(x,y,'k')
    #ax.plot(x,p(x),"k--",label="_nolegend_")

    if Type == "hist2d":
        ax.hist2d(x,y, bins=100, color='k')

    if Type == "bar":
        ax.bar(x,y, color='k')

    ax.set_xlabel('%s'%xlabel, fontsize=16)
    ax.set_ylabel('%s'%ylabel, fontsize=16)
    # Setting the ticks and tick marks
    ticks = [ax.xaxis.get_minor_ticks(),
             ax.xaxis.get_major_ticks()]
    marks = [ax.get_xticklabels(),
            ax.get_yticklabels()]
    for k in range(0,len(ticks)):
        for tick in ticks[k]:
            tick.label.set_fontsize(16)
    for k in range(0,len(marks)):
        for mark in marks[k]:
            mark.set_size(fontsize=16)
            mark.set_rotation(s=15)
    fig.tight_layout()
    if name==None:
        pass
    else:
        fig.savefig('%s'%name)
    fig.show()



if __name__ == "__main__":

    pop = np.loadtxt("populations.dat", delimiter=" ")
    print(pop)
    x = np.linspace(1,len(pop),len(pop))

    #print(len(x))
    #print(len(pop))
    exit()
    quick_plot(x=x,y=pop, name="test.png")




