#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
from io import open
#try: # for pip >= 10
#    from pip._internal.req import parse_requirements
#except ImportError: # for pip <= 9.0.3
#    from pip.req import parse_requirements
import sys
from pip._internal.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
# We need two seperate requirement files due to the failure to install all at once.
requirements = parse_requirements('requirements.txt', session=False)
#install_reqs2 = parse_requirements('requirements2.txt', session=False)
#
## reqs is a list of requirement
## e.g. ['django==1.5.1', 'mezzanine==1.4.6']
requirements = list(requirements)
try:
    requirements = [str(ir.req) for ir in requirements]
except:
    requirements = [str(ir.requirement) for ir in requirements]
#reqs1 = [str(ir.req) for ir in install_reqs1]
#req_links1 = [str(ir.url) for ir in install_reqs1]
#
#reqs2 = [str(ir.req) for ir in install_reqs2]
#req_links2 = [str(ir.url) for ir in install_reqs2]

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), "r", encoding='utf-8') as f:
    long_description = f.read()

sys.path.append('BICePs_2.0/')

setup(
        name="biceps",
        version="2.0.0",
        description='BICePs',
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
        #'Programming Language :: Python :: 2.7',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        #"Operating System :: OS Independent"
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
        url="https://biceps.readthedocs.io/en/latest/index.html",
            project_urls={
                "Github": "https://github.com/vvoelz/biceps",
                "Documentation": "https://biceps.readthedocs.io/en/latest/index.html",
            },
        author='Robert M. Raddi, Yunhui Ge, Vincent A. Voelz',
        author_email='rraddi@temple.edu, yunhui.ge@gmail.com, vvoelz@gmail.com',
        license='MIT',
        #packages=exclude=['docs']),
        packages=find_packages(),
        #setup_requires=['numpy'],
        install_requires=requirements,
        #install_requires=[
        #    'numpy',
        #    'mdtraj',
        #    #'git+https://github.com/mdtraj/mdtraj.git'
        #    'pymbar==3.0.3'],
            #'mdtraj==1.9.4','pymbar==3.0.2'],
        # conda install -c conda-forge mdtraj

        python_requires='>=3.7',
        #extras_require={  # Optional
        #        'dev': ['check-manifest'],
        #        'test': ['coverage'],
        #    },
        #dependency_links=[
            #"git+https://github.com/username/repo.git@MyTag"

            #'git+https://github.com/mdtraj/mdtraj.git'
            #'https://github.com/mdtraj/mdtraj/archive/1.9.4.tar.gz',
            #'git+https://github.com/mdtraj/mdtraj.git@1.9.3'
            #'https://github.com/mdtraj/mdtraj/tarball/master#eggmdtraj-1.9.4',
            #'git+https://github.com/choderalab/pymbar.git@3.0.3'
        #    ]
        #include_package_data=True,
        #zip_safe=True)
        )



