import numpy as np

class HEX():

    def __init__(self, dim = 3, order_dofs = 1, order_integration = "full"):

        self.dim = dim

        if(dim == 3):
            self.dofel = 8
        else:
            self.dofel = 4

        self.GaussianQuadrature(order_integration)

        self.ShapeFunctions()

    def GaussianQuadrature(self, order_integration):
        # Gaussian Quadrature initially defined for 3D full-order integration
        if(order_integration == "full"):
            val = 1. / np.sqrt(3)
            self.nip = 8
            self.IP_X = [ [-val, -val, -val],
                          [-val, -val, val],
                          [-val, val, -val],
                          [-val, val, val],
                          [val, -val, -val],
                          [val, -val, val],
                          [val, val, -val],
                          [val, val, val] ]
            self.IP_W = np.ones((8,1))
        else: # Reduced Integration
            self.nip = 1
            self.IP_X  = [[0.0, 0.0, 0.0]]
            self.IP_W = 8

    def ShapeFunctions(self):

        # Initialise lists of shape functions and derivatives at each integration point
        self.N = []
        self.dNdu = []

        for ip in range(self.nip):

            xi = self.IP_X[ip][0]
            eta = self.IP_X[ip][1]
            mu = self.IP_X[ip][2]

            N = np.zeros(8)

            N[0]=(1.-xi)*(1.-eta)*(1.-mu)
            N[1]=(1.+xi)*(1.-eta)*(1.-mu)
            N[2]=(1.+xi)*(1.+eta)*(1.-mu)
            N[3]=(1.-xi)*(1.+eta)*(1.-mu)
            N[4]=(1.-xi)*(1.-eta)*(1.+mu)
            N[5]=(1.+xi)*(1.-eta)*(1.+mu)
            N[6]=(1.+xi)*(1.+eta)*(1.+mu)
            N[7]=(1.-xi)*(1.+eta)*(1.+mu)

            N *= 0.25

            self.N.append(N) # Amend shape functions to list of shape ShapeFunctions

            dN = np.zeros((8,3))

            # Derivative Shape Functions

            dN[0,0] = -0.125 * (1 - eta) * (1 - mu)
            dN[1,0] =  0.125 * (1 - eta) * (1 - mu)
            dN[2,0] =  0.125 * (1 + eta) * (1 - mu)
            dN[3,0] = -0.125 * (1 + eta) * (1 - mu)
            dN[4,0] = -0.125 * (1 - eta) * (1 + mu)
            dN[5,0] =  0.125 * (1 - eta) * (1 + mu)
            dN[6,0] =  0.125 * (1 + eta) * (1 + mu)
            dN[7,0] = -0.125 * (1 + eta) * (1 + mu)

            dN[0,1]= -0.125 * (1 - xi) * (1 - mu);
            dN[1,1] = -0.125 * (1 + xi) * (1 - mu);
            dN[2,1] =  0.125 * (1 + xi) * (1 - mu);
            dN[3,1] =  0.125 * (1 - xi) * (1 - mu);
            dN[4,1] = -0.125 * (1 - xi) * (1 + mu);
            dN[5,1] = -0.125 * (1 + xi) * (1 + mu);
            dN[6,1] =  0.125 * (1 + xi) * (1 + mu);
            dN[7,1] =  0.125 * (1 - xi) * (1 + mu);

            dN[0,2] = -0.125 * (1 - xi) * (1 - eta);
            dN[1,2]  = -0.125 * (1 + xi) * (1 - eta);
            dN[2,2]  = -0.125 * (1 + xi) * (1 + eta);
            dN[3,2]  = -0.125 * (1 - xi) * (1 + eta);
            dN[4,2]  =  0.125 * (1 - xi) * (1 - eta);
            dN[5,2]  =  0.125 * (1 + xi) * (1 - eta);
            dN[6,2]  =  0.125 * (1 + xi) * (1 + eta);
            dN[7,2] =  0.125 * (1 - xi) * (1 + eta);

            self.dNdu.append(dN)
