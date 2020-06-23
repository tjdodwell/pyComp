from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np

from numpy.linalg import inv

from pyComp import *

comm = mpi.COMM_WORLD

class Cantilever():

    def __init__(self, param, comm):

        self.dim = 3

        self.comm = comm

        self.param = param

        self.numPlies = 1

        self.numInterfaces = self.numPlies - 1

        self.numLayers = self.numPlies + self.numInterfaces

        self.t = np.asarray([0.2, 0.02, 0.2, 0.02, 0.2])

        self.theta = np.asarray([np.pi/4, -1.0, 0.0, -1.0, 3.*np.pi/4])


        nx = 50
        ny = 10

        Lx = 10.
        Ly = 2.


        self.nel_per_layer = np.asarray([2,2,2,2,2])


        self.elementCutOffs = [0.0]


        for i in range(self.nel_per_layer.size):

            hz  = self.t[i] / self.nel_per_layer[i]

            for j in range(self.nel_per_layer[i]):
                self.elementCutOffs.append(self.elementCutOffs[-1] + hz)

        self.n = [nx, ny, np.sum(self.nel_per_layer) + 1] # Number of Nodes in each direction.

        self.L = [Lx, Ly, np.sum(self.nel_per_layer)] # Dimension in x - y plane are set, z dimension will be adjusted according to stacking sequence

        self.isBnd = lambda x: self.isBoundary(x)

        self.f = lambda x: self.rhs(x)

        self.da = PETSc.DMDA().create(self.n, dof=3, stencil_width=1)

        self.da.setUniformCoordinates(xmax=self.L[0], ymax=self.L[1], zmax=self.L[2])

        self.da.setMatType(PETSc.Mat.Type.AIJ)

        self.LayerCake() # Build layered composite from uniform mesh

        # Setup global and local matrices + communicators

        self.A = self.da.createMatrix()
        r, _ = self.A.getLGMap() # Get local to global mapping
        self.is_A = PETSc.IS().createGeneral(r.indices) # Create Index Set for local indices
        A_local = self.A.createSubMatrices(self.is_A)[0] # Construct local submatrix on domain
        vglobal = self.da.createGlobalVec()
        vlocal = self.da.createLocalVec()
        self.scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, self.is_A)

        self.A_local = A_local

        # Setup elements
        self.fe = ElasticityQ1()

        self.setMaterialParameters(self.param)

        self.isotropic, self.composite = makeMaterials(self.param)

        print(self.composite)

        # Build load vector as same for all solves

        #self.b = buildRHS(da, h, rhs)

    def isBoundary(self, x):
        vals = [0.0, 0.0, 0.0]
        output = False
        dofs = None
        if(x[0] < 1e-6):
            output = True
            dofs = [0, 1, 2]

        return output, dofs, vals

    def rhs(self, x):
        output = np.zeros((3,))
        output[2] = -9.81 * 0.0001
        return output

    def setTheta(self, angles):

        assert angles.shape[0] == self.numPlies, "Length of angles is not equal to number of plies"

        self.theta[0:2:-1] = angles

    def setMaterialParameters(self, param):

        self.E_R = param[0]
        self.nu_R = param[1]

        self.E1 = param[2]
        self.E2 = param[3]
        self.E3 = param[4]

        self.nu21 = param[5]
        self.nu31 = param[6]
        self.nu32 = param[7]

        self.G12 = param[8]
        self.G23 = param[9]
        self.G13 = param[10]

    def whichLayer(self, z):
        flag = False
        ans = 0
        for i in range(1,self.numLayers):
            if((z < self.cutoff[i]) and flag == False):
                ans = i
                flag = True
        return ans

    def LayerCake(self, plotMesh = True):

        nnodes = int(self.da.getCoordinatesLocal()[:].size/3)

        c = self.da.getCoordinatesLocal()[:]

        self.cutoff = np.cumsum(self.t, dtype=float)

        cnew = self.da.getCoordinatesLocal().copy()

        for i in range(nnodes):
            cnew[3 * i + 2] = self.elementCutOffs[np.int(c[3 * i + 2])]

        self.da.setCoordinates(cnew) # Redefine coordinates in transformed state.

        if(plotMesh):
            x = self.da.createGlobalVec()
            viewer = PETSc.Viewer().createVTK('initial_layer_cake.vts', 'w', comm = PETSc.COMM_WORLD)
            x.view(viewer)
            viewer.destroy()

    def getIndices(self,elem, dof = 3):
        ind = np.empty(dof*elem.size, dtype=np.int32)
        for i in range(dof):
            ind[i::dof] = dof*elem + i
        return ind


    def solve(self, theta, plotSolution = False, filename = "solution"):

        # Solve A * x = b

        # Assemble Global Stifness Matrix

        self.A = self.da.createMatrix()

        b = self.da.createGlobalVec()
        b_local = self.da.createLocalVec()

        elem = self.da.getElements()
        nnodes = int(self.da.getCoordinatesLocal()[ :].size/self.dim)
        coords = np.transpose(self.da.getCoordinatesLocal()[:].reshape((nnodes,self.dim)))
        for ie, e in enumerate(elem,0): # Loop over all local elements
            midpoint_z = np.mean(coords[2,e])
            layerId = np.int(self.whichLayer(midpoint_z))
            isComposite = False
            if(self.theta[layerId] >= 0):
                C = self.composite
                isComposite = True
                angle = self.theta[layerId]
            else:
                C = self.isotropic
                angle = None

            Ke = self.fe.getLocalStiffness(coords[:,e], C, isComposite, angle)

            ind = self.getIndices(e)


            self.A.setValuesLocal(ind, ind, Ke, PETSc.InsertMode.ADD_VALUES)
            b_local[ind] = self.fe.getLoadVec(coords[:,e],  self.f)

        self.A.assemble()
        self.comm.barrier()

        # Implement Boundary Conditions
        rows = []
        for i in range(nnodes):
            flag, dofs, vals = self.isBoundary(coords[:,i])
            if(flag): # It's Dirichlet
                for j in range(len(dofs)): # For each of the constrained dofs
                    index = 3 * i + dofs[j]
                    rows.append(index)
                    b_local[index] = vals[j]
        rows = np.asarray(rows,dtype=np.int32)

        self.A.zeroRowsLocal(rows, diag = 1.0)

        self.scatter_l2g(b_local, b, PETSc.InsertMode.INSERT_VALUES)

        # Solve

        # Setup linear system vectors
        x = self.da.createGlobalVec()
        x.setRandom()
        xnorm = b.dot(x)/x.dot(self.A*x)
        x *= xnorm

        # Setup Krylov solver - currently using AMG
        ksp = PETSc.KSP().create()
        pc = ksp.getPC()
        ksp.setType('cg')
        pc.setType('gamg')

        # Iteratively solve linear system of equations A*x=b
        ksp.setOperators(self.A)
        ksp.setInitialGuessNonzero(True)
        ksp.setFromOptions()
        ksp.solve(b, x)

        if(plotSolution): # Plot solution to vtk file
            viewer = PETSc.Viewer().createVTK(filename + ".vts", 'w', comm = comm)
            x.view(viewer)
            viewer.destroy()

        # Post process all quantities of interest

        Q = 1.0 # To do!

        return Q




param = [ None ] * 11

param[0] = 4.5  # E_R   GPa
param[1] = 0.35 # nu_R

param[2] = 135  # E1    GPa
param[3] = 8.5  # E2    GPa
param[4] = 8.5  # E3    GPa

param[5] = 0.022    # nu_21
param[6] = 0.022    # nu_31
param[7] = 0.45     # nu_32

param[8] = 5.0  # G_12 GPa
param[9] = 5;   # G_13 GPa
param[10] = 5;  # G_23 GPa

myModel = Cantilever(param, comm)

myModel.solve(None, True)
