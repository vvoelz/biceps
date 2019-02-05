from setuptools import setup

setup(
    name="BICePs",
    version="2.0",
    install_requires=[
        "mdtraj",
        "pymbar",
    ],
    dependency_links = [
        "https://github.com/mdtraj/mdtraj/archive/master.zip",
        "https://github.com/choderalab/pymbar/archive/master.zip"
        ]
)
