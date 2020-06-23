import numpy as np

import pytest

from pyComp import *

from petsc4py import PETSc

@pytest.fixture
def defineTestMaterial():
    '''Define '''

    param = [ None ] * 11

    param[0] = 4.5  # E_R   GPa
    param[1] = 0.35 # nu_R

    param[2] = 135  # E1    GPa
    param[3] = 8.5  # E2    GPa
    param[4] = 8.5  # E3    GPa

    param[5] = 0.022    # nu_21
    param[6] = 0.022    # nu_31
    param[7] = 0.5     # nu_32

    param[8] = 5.0  # G_12 GPa
    param[9] = 5;   # G_13 GPa
    param[10] = 5;  # G_23 GPa

    return param

def test_tensor_rotation(defineTestMaterial):

    param = defineTestMaterial

    isotropic, composite = makeMaterials(defineTestMaterial)

    fe = ElasticityQ1()

    testC = fe.RotateFourthOrderTensor(isotropic, 0.234)

    assert np.sum(np.abs(isotropic - testC)) < 1e-3

    testC = fe.RotateFourthOrderTensor(composite, np.pi)

    assert np.sum(np.abs(composite - testC)) < 1e-3

def test_make_Q1_FEM(defineTestMaterial):

    param = defineTestMaterial

    isotropic, composite = makeMaterials(defineTestMaterial)

    fe = ElasticityQ1()
