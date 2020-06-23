from __future__ import print_function, division
import sys, petsc4py
petsc4py.init(sys.argv)
import mpi4py.MPI as mpi
from petsc4py import PETSc
import numpy as np

from numpy.linalg import inv

comm = mpi.COMM_WORLD

class Cantilever():

    def __init__(self, comm):

        self.dim = 3

        self.comm = comm

        self.numPlies = 2

        self.numInterfaces = self.numPlies - 1

        self.numLayers = self.numPlies + self.numInterfaces

        #self.t = np.asarray([0.2, 0.02, 0.2])

        self.t = np.asarray([0.2, 0.02, 0.2, 0.02, 0.2, 0.02, 0.2, 0.02, 0.2])

        #self.theta = np.asarray([0., -1., 0.])

        self.theta = np.asarray([0., -1., 0., -1., 0., -1., 0., -1., 0])

        nx = 20
        ny = 6

        Lx = 100.
        Ly = 20.

        self.nel_per_layer = np.asarray([2,1,2,1,2,1,2,1,2])

        self.elementCutOffs = [0.0]

        for i in range(self.nel_per_layer.size):

            hz  = self.t[i] / self.nel_per_layer[i]

            for j in range(self.nel_per_layer[i]):
                self.elementCutOffs.append(self.elementCutOffs[-1] + hz)

        self.n = [nx, ny, np.sum(self.nel_per_layer) + 1] # Number of Nodes in each direction.

        self.L = [Lx, Ly, np.sum(self.nel_per_layer)] # Dimension in x - y plane are set, z dimension will be adjusted according to stacking sequence

        self.isBnd = lambda x: self.isBoundary(x)

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


        # Build load vector as same for all solves

        #self.b = buildRHS(da, h, rhs)

    def isBoundary(self, x):
        val = 0.0
        output = False
        if(x[0] < 1e-6):
            output = True
            dofs = [0, 1, 2]
            vals = [0.0, 0.0, 0.0]

        return output, dofs, val

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

    def makeMaterials(self):

        # Isotropic Resin
        lam = self.E_R * self.nu_R/((1+self.nu_R)*(1-2*self.nu_R))
        mu = self.E_R/(2.*(1.+self.nu_R));
        self.isotropic = np.zeros((6,6));
        self.isotropic[0:3,0:3] = lam * np.ones((3,3));
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
            print("This is an element!")
            #Ke = self.fe.getLocalStiffness(coords[:,e], 1.0)
            #self.A.setValuesLocal(e, e, Ke, PETSc.InsertMode.ADD_VALUES)
            #b_local[e] = self.fe.getLoadVec(coords[:,e])
        self.A.assemble()
        self.comm.barrier()

        # Solve

        # Setup linear system vectors
        x = self.da.createGlobalVec()
        x.setRandom()
        #xnorm = b.dot(x)/x.dot(self.A*x)
        #x *= xnorm

        if(plotSolution): # Plot solution to vtk file
            viewer = PETSc.Viewer().createVTK(filename + ".vts", 'w', comm = comm)
            x.view(viewer)
            viewer.destroy()

        # Post process all quantities of interest

        Q = 1.0

        return Q


print("Yer boi")


myModel = Cantilever(comm)

myModel.solve(None, True)
