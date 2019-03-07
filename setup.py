#!/usr/bin/env python

from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
import sys

# parse_requirements() returns generator of pip.req.InstallRequirement objects
# We need two seperate requirement files due to the failure to install all at once.
#install_reqs1 = parse_requirements('./doc/requirements1.txt', session=False)

install_reqs2 = parse_requirements('./doc/requirements2.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
#reqs1 = [str(ir.req) for ir in install_reqs1]
#req_links1 = [str(ir.url) for ir in install_reqs1]

reqs2 = [str(ir.req) for ir in install_reqs2]
req_links2 = [str(ir.url) for ir in install_reqs2]



setup(
        name="BICePs",
        version="2.0",
        description='BICePs',
        long_description='This is BICePs',
        classifiers=[
        #'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Science :: Engineering'],
        #keywords='',
        #url='http://github.com/',
        #author='',
        #author_email='',
        license='MIT',
        #packages=['BIcePs'],
        packages=find_packages(),
        #install_requires=[reqs1,reqs2],
        install_requires=reqs2,
        #dependency_links=[req_links1,req_links2],
        dependency_links=req_links2,
        include_package_data=True,
        zip_safe=False)



