
import numpy as np

from .HEX import HEX

class ElasticityQ1(HEX):

    def __init__(self, dim = 3, order_integration = "full"):

        super().__init__(dim, 1, order_integration)

        self.dofel = 24

    def getLocalStiffness(self, x, C, isAnisotropic = False, e, n):
        Ke = np.zero((self.dofel, self.dofel))
        for ip in range(self.nip):
            J = np.matmul(x, self.dNdu[ip])
            dNdX = np.matmul(self.dNdu[ip],np.linalg.inv(J))
            # For Composite Elements
            if(isComposite):
                hatC = self.RotateFourthOrderTensor(C, e)

            B = np.zeros((6, 24))

            B(0,0:8) = dNdX(:,0); # e_11 = u_1,1
            B(1,8:16) = dNdX(:,1); # e_22 = u_2,2
            B(2,16:24) = dNdX(:,2); # e_33 = u_3,3

            B(3,8:16) = dNdX(:,2);	B(3,16:24) = dNdX(:,1);	# e_23 = u_2,3 + u_3,2
            B(4,0:8) = dNdX(:,2);	B(4,16:24) = dNdX(:,0);	# e_13 = u_1,3 + u_3,1
            B(5,0:8) = dNdX(:,1);	B(5,8:16) = dNdX(:,0);	# e_12 = u_1,2 + u_2,1

            Ke += np.matmul(np.transpose(B),np.matmul(hatC,B)) * np.linalg.det(J) * self.IP_W[ip]

        return Ke

    def getLoadVec(self, x, func ):
        # getRHS - computes load vector
        # x - coordinates of element nodes np.array - dim by nodel
        # func - function handle to evaluate source function at given x in Omega
        fe = np.zeros(24)
        for ip in range(self.nip):
            J = np.matmul(x, self.dNdu[ip])
            x_ip = np.matmul(x,self.N[ip])
            val = func(x_ip)
            for i in range(3):
                fe[i*8:(i+1)*8] = fe[i*8:(i+1)*8] + val[i] * self.N[ip] * np.linalg.det(J) * self.IP_W[ip]

        return fe

    def RotateFourthOrderTensor(self, C, theta)

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

        Chat = np.matmult(R,np.matmul(C,np.transpose(R)));

        return Chat
