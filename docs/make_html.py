
import subprocess,os
source = "./"
outdir = "OUT"
if os.path.exists(outdir): subprocess.Popen(f"rm -r {outdir}", shell=True)
if os.path.exists(outdir): subprocess.Popen(f"rm -r api", shell=True)
subprocess.Popen(f"sphinx-build -b html {source} {outdir}", shell=True)


# Other scripts and things:{{{
''' NOTES from other scripts.
echo "Building biceps..."
$PYTHON setup.py install     # Python command to install the script.


#!/usr/bin bash

printf 'This is a script to build the html files, then open up a webserver to view the docs'

# Define the Project Name and the Authors of this Project
Proj_Name='biceps'

wd=$PWD

## Check to see if python can see your package:
python -c 'import '$Proj_Name;

# Set the Python path only for the build command
#FIXME:
PYTHONPATH=$wd/$Proj_Name make html;

# Now, we want to see what we have created so far...
# Open up a simple python webserver.
open http://localhost:8000/
python -m SimpleHTTPServer;


'''
# }}}



