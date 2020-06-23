
import numpy as np

from .HEX import HEX

class ElasticityQ1(HEX):

    def __init__(self, dim = 3, order_integration = "full"):

        super().__init__(dim, 1, order_integration)

        self.dofel = 24

        self.index_u = 3 * np.arange(8)
        self.index_v = 3 * np.arange(8) + 1
        self.index_w = 3 * np.arange(8) + 2

    def getLocalStiffness(self, x, C, isComposite, angle):

        Ke = np.zeros((self.dofel, self.dofel))

        for ip in range(self.nip):

            J = np.matmul(x, self.dNdu[ip])

            dNdX = np.matmul(self.dNdu[ip],np.linalg.inv(J))

            # For Composite Elements
            if(isComposite):
                hatC = self.RotateFourthOrderTensor(C, angle)
            else:
                hatC = C

            B = self.computeBMatrix(dNdX)

            Ke += np.matmul(np.transpose(B),np.matmul(hatC,B)) * np.linalg.det(J) * self.IP_W[ip]
        return Ke

    def getLoadVec(self, x, func):
        # getRHS - computes load vector
        # x - coordinates of element nodes np.array - dim by nodel
        # func - function handle to evaluate source function at given x in Omega
        fe = np.zeros((8,3))
        for ip in range(self.nip):
            J = np.matmul(x, self.dNdu[ip])
            x_ip = np.matmul(x,self.N[ip])
            val = func(x_ip)
            for i in range(3):
                fe[0:8,i] = fe[0:8,i] + val[i] * self.N[ip] * np.linalg.det(J) * self.IP_W[ip]

        return fe.reshape(24)


    def computeBMatrix(self,dNdX):

        B = np.zeros((6, 24))

        B[0,self.index_u] = dNdX[:,0]; # e_11 = u_1,1
        B[1,self.index_v] = dNdX[:,1]; # e_22 = u_2,2
        B[2,self.index_w] = dNdX[:,2]; # e_33 = u_3,3

        B[3,self.index_v] = dNdX[:,2];	B[3,self.index_w] = dNdX[:,1];	# e_23 = u_2,3 + u_3,2
        B[4,self.index_u] = dNdX[:,2];	B[4,self.index_w] = dNdX[:,0];	# e_13 = u_1,3 + u_3,1
        B[5,self.index_u] = dNdX[:,1];	B[5,self.index_v] = dNdX[:,0];	# e_12 = u_1,2 + u_2,1

        return B

    def RotateFourthOrderTensor(self, C, theta):

        # Rotation about x_3
        c = np.cos(theta); s = np.sin(theta);
        R = np.zeros((6,6))
        R[0,0] = c * c
        R[0,1] = s * s;
        R[0,5] = 2 * c * s;
        R[1,0] = s * s;
        R[1,1] = c * c;
        R[1,5] = -2 * c * s;
        R[2,2] = 1;
        R[3,3] = c;
        R[3,4] = s;
        R[4,3] = -s;
        R[4,4] = c;
        R[5,0] = -c * s;
        R[5,1] = c * s;
        R[5,5] = (c * c) - (s * s);

        Chat = np.matmul(R,np.matmul(C,np.transpose(R)));

        return Chat

    def getIndices(self,elem, i, dof = 3):
        ind = np.empty(elem.size, dtype=np.int32)

        ind[i::dof] = 3*elem + i
        return ind
