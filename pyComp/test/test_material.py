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

def test_material_tensors(defineTestMaterial):

    isotropic, composite = makeMaterials(defineTestMaterial)

    IsoTrue = np.asarray([[ 7.2222,3.8889,3.8889,0.,0.,0.],
    [3.8889,    7.2222,    3.8889 ,        0 ,        0  ,       0],
    [3.8889,    3.8889   ,   7.2222  ,       0     ,    0  ,       0],
    [     0    ,     0    ,     0   , 1.6667    ,     0    ,     0],
    [ 0     ,    0       ,  0     ,    0  ,  1.6667      ,   0],
    [     0    ,     0    ,     0    ,     0      ,   0  ,  1.6667]])

    CompTrue = np.asarray([ [139.2827,    6.1284,    6.1284,         0,         0,         0],
    [6.1284  , 11.6030 ,   5.9363   ,      0      ,   0    ,     0],
    [6.1284  ,  5.9363  , 11.6030    ,     0      ,   0     ,    0],
    [0    ,     0    ,     0   , 5.0000   ,      0     ,    0],
    [0    ,     0     ,    0   ,      0   , 5.0000   ,     0],
    [0      ,   0    ,     0    ,     0     ,    0  ,  5.0000]])

    assert np.sum(np.abs(isotropic - IsoTrue)) < 1e-3

    assert np.sum(np.abs(composite - CompTrue)) < 1e-3
