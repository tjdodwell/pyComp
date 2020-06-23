import numpy as np

from numpy.linalg import inv


def makeMaterials(param):

    E_R = param[0]
    nu_R = param[1]

    E1 = param[2]
    E2 = param[3]
    E3 = param[4]

    nu21 = param[5]
    nu31 = param[6]
    nu32 = param[7]

    G12 = param[8]
    G23 = param[9]
    G13 = param[10]

    # Isotropic Resin
    lam = E_R * nu_R/((1+nu_R)*(1-2*nu_R))
    mu = E_R/(2.*(1.+nu_R));
    isotropic = np.zeros((6,6));
    isotropic[0:3,0:3] = lam * np.ones((3,3));
    for i in range(6):
        if(i < 3):
            isotropic[i,i] += 2.0 * mu
        else:
            isotropic[i,i] += mu

    # Orthotropic Composite
    S = np.zeros((6,6));
    S[0,0] = 1/E1
    S[0,1] = -nu21/E2
    S[0,2] = -nu31/E3;
    S[1,0] = S[0,1]
    S[1,1] = 1/E2
    S[1,2] = -nu32/E3;
    S[2,0] = S[0,2]
    S[2,1] = S[1,2]
    S[2,2] = 1/E3;
    S[3,3] = 1/G23;
    S[4,4] = 1/G13;
    S[5,5] = 1/G12;

    composite = inv(S)

    return isotropic, composite
