import mpi4py.MPI as mpi

from petsc4py import PETSc

import numpy as np

def LayerCake(nx, ny, Lx, Ly, t, nel_per_layer, plotMesh = True, overlap = 1):

    n = [nx, ny, np.sum(nel_per_layer) + 1] # Number of Nodes in each direction.

    L = [Lx, Ly, np.sum(nel_per_layer)] # Dimension in x - y plane are set, z dimension will be adjusted according to stacking sequence

    da = PETSc.DMDA().create(n, dof=3, stencil_width=overlap)

    da.setUniformCoordinates(xmax=L[0], ymax=L[1], zmax=L[2])

    da.setMatType(PETSc.Mat.Type.AIJ)

    elementCutOffs = [0.0]

    for i in range(nel_per_layer.size):
        hz  = t[i] / nel_per_layer[i]
        for j in range(nel_per_layer[i]):
            elementCutOffs.append(elementCutOffs[-1] + hz)

    nnodes = int(da.getCoordinatesLocal()[:].size/3)

    c = da.getCoordinatesLocal()[:]

    cnew = da.getCoordinatesLocal().copy()

    for i in range(nnodes):
        cnew[3 * i + 2] = elementCutOffs[np.int(c[3 * i + 2])]

    da.setCoordinates(cnew) # Redefine coordinates in transformed state.

    if(plotMesh):
        x = da.createGlobalVec()
        viewer = PETSc.Viewer().createVTK('initial_Geometry.vts', 'w', comm = PETSc.COMM_WORLD)
        x.view(viewer)
        viewer.destroy()

    return da
