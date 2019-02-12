#!/usr/bin/env python

from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
import sys

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('./doc/source/requirements.txt')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]
req_links = [str(ir.url) for ir in install_reqs]


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
        install_requires=reqs,
        dependency_links=req_links,
        include_package_data=True,
        zip_safe=False)



