### Testing script to be used with
### $ pytest -v

import os, sys

# Build the notebook into a script that we can test 
os.system('jupyter nbconvert --to python albocycline.ipynb')
# ... this should output albocycline.py

import albocycline 

# test_albocycline()

