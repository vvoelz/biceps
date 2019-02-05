from setuptools import setup

setup(
    name="BICePs",
    version="2.0",
    install_requires=[
        "numpy",
        "cython".
        "mdtraj>=1.9",
        "pymbar"
    ],
    #dependency_links = [
    #    "https://github.com/mdtraj/mdtraj/archive/master.zip",
    #    "https://github.com/choderalab/pymbar/archive/master.zip"
    #    ]
)
