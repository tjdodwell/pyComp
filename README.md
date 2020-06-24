# pyComp

This is a lightweight parallel implementation of the finite element method (FEM) for solving composite structural problems. The code is built on the library [petsc4py](https://bitbucket.org/petsc/petsc4py/src/master/) and mpi4py. Codes run in both sequential and parallel modes.

Code has been developed for solving simple mechanics problems to support the development of new Bayesian methodologies within a larger EPSRC programme grant - called [CERTEST](https://www.composites-certest.com).

**This is a new code, testing is being gradually added, please use with a very healthy level of scepticism**

## Installation - tested on Ubuntu and macOS Mojave 10.14.1

To install this project you must first clone the project

```
git clone https://github.com/tjdodwell/pyComp.git
cd pyComp
```

Next is to create an environment with all the required packages

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

All unit tests can be run by calling

```
pytest
```

in the main directory

## Basic Example - Cantilever Beam

A simple example of a flat composite laminate made up of 3 layers and 2 interfaces, clamped at one end, and deforming under self weight is provided as a basic example. To run this example (sequentially)

```
cd examples
python cantilever.py -ksp_monitor
```

The output to screen will show the iterations of the iterative solver. A `solution.vts` file will be generate. This can be opened using the open source software [paraview](https://www.paraview.org).

![alt text](https://github.com/tjdodwell/pyComp/docs/master/cantilever.png?raw=true)

