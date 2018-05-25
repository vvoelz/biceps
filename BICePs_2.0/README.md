# BICePs 2.0:
## We are going to give a rebirth for BICePs. Here are some points we want to achieve:
### 1. there will be three phases of new BICePs: 1/ input file generation 2/ sampling by BICePs 3/ analysis using MBAR and plot figures if applicable.
### 1/ input file generation
To get the new BICePs work well, we need to prepare input file with a format that BICePs can read. To do this, we need to create scripts for people that can be easily used.

### 2/ sampling by BICePs
a/ We need to make sure BICePs can work fast and light enough. We need to prevent any unnecessary work done by BICePs. It was an issue in the past. For example, we don't need BICePs sampling PF or noe but it's still do it for some reason. Also we are saving lists of allowed/sampled sigma_cs_H to the results even we don't need them at all. Those are problems of the old version so we need to make sure they are fixed in the new version.
b/ We need to reorganize the scripts and make it easier to add some new class or functions for new exerpimetnal observables. To do this we need to reduce numerb of functions as much as possible so that we have some universal functions that can work for every observables. Once we have new observables we don't need to touch the source code or only a tiny part need to be modified. 
c/ We need to be smart in this case and everytime we need to think if there is a better way to do it to make BICePs faster and lower the memory usage. We need to consider complexity in the space and time both.

### 3/ analysis
We rely on MBAR a lot in our analysis so make sure BICePs can work different PyMBAR version is necessary. We need to be clear what version PyMBAR is working well with BICePs. A highly automatic scripts for plotting is also helpful. Combined them together will reduce the time and memory usage a lot.

### After the scripts are done, we need to prepare tutorials, build the website and get it work for anaconda/pip installation. Also, creating a new github repository for BICePs 2.0 is necessary.


