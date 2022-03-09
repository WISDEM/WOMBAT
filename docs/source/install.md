# Installation


## Create a conda environoment

Download the latest version of [Miniconda](<https://docs.conda.io/en/latest/miniconda.html>)
for the appropriate OS. Follow the remaining [steps](<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>)
for the appropriate OS version.


Using conda, create a new virtual environment, replacing `<environment_name>` with a name
of your choosing (without spaces):
```text
$ conda create -n <environment_name> python=3.7 --no-default-packages
$ conda activate <environment_name>
$ conda install -c anaconda pip  # installing Python above also install pip
```
You can now use ``conda activate <environment_name>`` to enter the environment and
``conda deactivate`` to exit the environment.


## Installing

```{note}
The folowing all asssume you are in your `wombat` environment!
```


### Pip

```text
$ pip install wombat
```

### From Source

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ pip install wombat/
```


### For contributors:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ pip install -e '.[dev]'
# Required for automatic code formatting!
$ pre-commit install
```


### For documentation:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ pip install -e '.[docs]'
```

#### Build the documentation site

```text
$ cd docs/
$ sphinx-build -b html source _build
$ make html
```

Now, the documentation site should be able to be viewed locally at
docs/_build/html/index.html
