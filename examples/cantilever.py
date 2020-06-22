from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np

from numpy.linalg import inv

class Cantilever():

    def __init__(self):

        self.numPlies = 5

        self.numInterfaces = self.numPlies - 1

        self.t = np.asarray([0.2, 0.02, 0.2, 0.02, 0.2, 0.02, 0.2, 0.02, 0.2])

        self.theta = np.asarray([0., -1., 0., -1., 0., -1., 0., -1., 0])

        self.nel_per_layer = np.asarray([4,2,4,2,4,2,4,2,4])

        self.n = [10, 10, np.sum(self.nel_per_layer)] # Number of Nodes in each direction.

        self.L = [20., 100., np.sum(self.t)] # Dimension in x - y plane are set, z dimension will be adjusted according to stacking sequence

        self.da = PETSc.DMDA().create([n[0], n[1], n[2]], dof=3, stencil_width=1)

        self.LayerCake() # Build layered composite from uniform mesh

        #da.setMatType(PETSc.Mat.Type.IS)

        # Build load vector as same for all solves

        #self.b = buildRHS(da, h, rhs)



    def setTheta(self, angles):

        assert angles.shape[0] == self.numPlies, "Length of angles is not equal to number of plies"

        self.theta[0:2:-1] = angles

    def setMaterialParameters(self, param):

        self.E_R = param[0]
        self.nu_R = param[1]

        self.E1 = param[2]
        self.E2 = param[3]
        self.E3 = param[4]

        self.nu12 = param[5]
        self.nu23 = param[6]
        self.nu13 = param[7]

        self.G12 = param[8]
        self.G23 = param[9]
        self.G13 = param[10]

    def whichLayer(self, x):
        flag = False
        ans = 0.0
        for i in range(self.numLayers):
            if((x[2] < self.cutoff[i]) and flag == False):
                ans = k
                flag = True
        return k

    def LayerCake(self):

        uniform_coords = self.da.getCoordinates()

        self.cutoff = np.cumsum(self.t, dtype=float)

        coords = np.zeros(uniform_coords.shape)

        coords[0,:] = self.Lx * uniform_coords[0,:]
        coords[1,:] = self.Ly * uniform_coords[1,:]

        coords = np.zeros(uniform_coords.shape)

        coords[0,:] = self.Lx * uniform_coords[0,:]
        coords[1,:] = self.Ly * uniform_coords[1,:]

        for i in range(uniform_coords.shape[1]):

            id = self.whichLayer(uniform_coords[2,i]) # Which Layer

            hz = (self.cutoff[id + 1] - self.cutoff[id]) / nelz[id]

            j = np.floor((uniform_coords[2,i] - self.cutoff[id]) / hz)

            coords[2,i] = self.cutoff[id] + j * hz

        self.da.setCoordinate(coords) # Redefine coordinates in transformed state.

    def makeMaterials(self):

        # Isotropic Resin
        lam = self.E_R * self.nu_R/((1+self.nu_R)*(1-2*self.nu_R))
        mu = self.E_R/(2.*(1.+self.nu_R));
        self.isotropic = np.zeros((6,6));
        self.isotropic(0:3,0:3) = lam * np.ones((3,3));
        for i in range(6):
            if(i < 3):
                self.isotropic[i,i] += 2.0 * mu
            else:
                self.isotropic[i,i] += mu

        # Orthotropic Composite
        S = np.zeros((6,6));
        S[0,0] = 1/self.E1
        S[0,1] = -self.nu_21/self.E2
        S[0,2] = -self.nu_31/self.E3;
        S[1,0] = S[0,1]
        S[1,1] = 1/self.E2
        S[1,2] = -self.nu_32/self.E3;
        S[2,0] = S[0,2]
        S[2,1] = S[1,2]
        S[2,2] = 1/self.E3;
        S[3,3] = 1/self.G_23;
        S[4,4] = 1/self.G_13;
        S[5,5] = 1/self.G_12;

        self.composite = inv(S)


    def solve(self, theta, plotSolution = False):

        A = buildElasticityMatrix(da, h, lamb, mu)

        A.assemble()

        b = self.b.copy() # Copy right hand side

        bcApplyWest(da, A, b) # Apply boundary conditions - updating A and b


        pcbnn = PCBNN(A)

        # Set initial guess
        x = da.createGlobalVec()
        x.setRandom()
        xnorm = b.dot(x)/x.dot(A*x)
        x *= xnorm

        # Setup linear solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType(ksp.Type.PYTHON)
        pyKSP = KSP_AMPCG(pcbnn)
        pyKSP.callback = callback(da)
        ksp.setPythonContext(pyKSP)
        ksp.setInitialGuessNonzero(True)
        ksp.setFromOptions()

        ksp.solve(b, x) # Solve

        if(plotSolution):

            viewer = PETSc.Viewer().createVTK('solution_3d_asm.vts', 'w', comm = PETSc.COMM_WORLD)
            x.view(viewer)
            viewer.destroy()

        # Post process all quantities of interest

        Q = 1.0

        return Q
