# pyComp

This is a lightweight (parallel) implementation of finite element method (FEM) for solving composite structural analysis. The code is built on the library [petsc4py](https://bitbucket.org/petsc/petsc4py/src/master/) and mpi4py. Codes run in both sequential and parallel modes.

Code has been developed for solving simple mechanics problems to support development of new Bayesian methodologies with a larger EPSRC programme grant - called [CERTEST](https://www.composites-certest.com).

** This is a new code, testing is being gradually added, please use with a very healthy level of scepticism **

## Installation

To install this project you must first clone the project

```
git clone https://github.com/tjdodwell/pyComp.git
cd pyComp
```

Next is to create a package with all the required packages

```
conda env create -f environment.yml
```

The environment is then activated

```
source activate pyComp
```

The package can be installed by running setup.py

```
python setup.py install
```
