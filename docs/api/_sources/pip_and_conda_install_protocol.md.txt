# Protocol for setting up `pip install`

Please see:
#### Tutorial for [Packing Python Projects](https://packaging.python.org/tutorials/packaging-projects/)


##### Prerequisites:

```bash
pip install --user --upgrade setuptools wheel

pip install --user --upgrade twine

```

--------------------------------------

1. Register an account on [https://pypi.org](https://pypi.org/) or [https://test.pypi.org](https://test.pypi.org/) if you would like to do testing first.

2. Create a `setup.py` - main commands/controls for the project
to be built.

3. Run:

```
python setup.py sdist bdist_wheel
```

 - The command that is used to build the distribution archives (The `tar.gz` file is a source archive whereas the `.whl` file is a built distribution)


4. Lastly, run the following command to upload to your pypi account:

```
twine upload dist/*
```




# Protocol for setting up `conda install`

Please see:
#### Tutorial for [Uploading Conda Packages](https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-packages/#uploading-conda-packages)


##### Prerequisites:

```bash
conda install anaconda-client conda-build

```

--------------------------------------

1. Register an account on [https://anaconda.org/](https://anaconda.org/).

2. Create a `meta.yaml` - main commands/controls for the project
to be built.

3. Run `conda build` in the directory of `meta.yml`

```
conda build .
```

4. Lastly, login to your anaconda account and upload the package

```
anaconda login
anaconda upload /path/to/conda-package.tar.bz2
```







