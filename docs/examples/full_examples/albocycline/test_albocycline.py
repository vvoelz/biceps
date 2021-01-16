### Testing script to be used with
### $ pytest -v

import os, sys

# Build the notebook into a script that we can test 
os.system('jupyter nbconvert --to python albocycline.ipynb')
# ... this should output albocycline.py


def test_albocycline():
    """Tests the code in albocycline.py by importing it.""" 
    import albocycline 

test_albocycline()

