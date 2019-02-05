#!/usr/bin/env python

from setuptools import setup, find_packages

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
        install_requires=["numpy","cython","mdtraj","pymbar"],
        include_package_data=True,
        zip_safe=False)
