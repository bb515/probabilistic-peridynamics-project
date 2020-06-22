# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:27:01 2020

@author: Tim Dodwell
"""

import numpy as np

class Grid:

    def __init__(self):

        self.dim = 2
        self.nel = 0
        self.numNodes = 0

    def findNeighbours(self):
        # Simple implementation for now as coarse mesh will be small
        M = np.zeros((self.numNodes, self.numNodes), dtype = int)

        for ie in range(0, self.nel):
            nodes = self.connectivity[ie][:]
            for i in range(0, 4):
                for j in range(0,4):
                    if(i != j):
                        M[nodes[i]][nodes[j]] = 1
        # Build List of Neighbours
        self.neighbours = [ [] for i in range(self.numNodes) ]
        for i in range(0, self.numNodes):
            for j in range(0, self.numNodes):
                if (M[i][j] == 1):
                    self.neighbours[i].append(int(j))

        # Build list of macroscale elements in which a node lives
        tmpN2E = [ [] for i in range(self.numNodes) ]
        for ie in range(0, self.nel): # For each elements
            nodes = self.connectivity[ie][:]
            for j in range(0,nodes.size):
                tmpN2E[nodes[j]].append(ie)

        self.Node2Elements = [ [] for i in range(self.numNodes) ]

        for i in range(0, self.numNodes):
            tmpArray = np.array(tmpN2E[i])
            self.Node2Elements[i] = np.unique(tmpArray)


        return self.neighbours, self.Node2Elements

    def buildStructuredMesh2D(self,L,n,X0,order,verb = 2):

        self.X0 = X0

        self.L = L

        if(verb > 0):
            print('Building Structured 2D Grid!')

        self.n = n

        # Function builds a structured finite element mesh in 2D
        if(order == 1):
            numNodesX = n[0] + 1
            numNodesY = n[1] + 1
            self.nodePerElement = 4
            self.numNodes = numNodesX * numNodesY
            self.nel = n[0] * n[1]

        else:
            print('Grids of order 2 or high are not currently supported, assuming order 1')
            order == 1;

        x = np.linspace(X0[0], X0[0] + L[0], numNodesX)
        y = np.linspace(X0[1], X0[1] + L[1], numNodesY)

        self.h = np.zeros(self.dim)

        for i in range(0,self.dim):
            self.h[i] = L[i] / n[i]

        # Nodes will be formed from a tensor product of this two vectors
        self.coords = np.zeros((self.numNodes, 2))


        count = 0
        for i in range(0, n[1] + 1):
            for j in range(0, n[0] + 1):
                self.coords[count][0] = x[j]
                self.coords[count][1] = y[i]
                count += 1 # increment node counter

        # Build Connectivity Matrix
        self.nel = n[0] * n[1]
        self.connectivity = np.zeros((self.nel, self.nodePerElement), dtype = int)
        count = 0
        ncount = 0
        for j in range(0, n[1]):
            for i in range(0, n[0]):
                self.connectivity[count][0] = ncount
                self.connectivity[count][1] = ncount + 1
                self.connectivity[count][2] = ncount + numNodesX + 1
                self.connectivity[count][3] = ncount + numNodesX
                count += 1 # increment element counter
                ncount += 1

            ncount+=1

        if(verb > 1):
            print('... Grid Built!')
            print('Number of Nodes ' + str(self.numNodes))
            print('Number of Elements ' + str(self.nel))
            for i in range(0, self.numNodes):
                print(str(self.coords[i][:]))
            for i in range(0, self.nel):
                print(str(self.connectivity[i][:]))

    def particletoCell_structured(self,pCoords):
        numParticles = int(pCoords[:].size / self.dim)
        p2e = np.zeros(numParticles, dtype = int)
        p_localCoords = np.zeros((numParticles, self.dim))

        self.e2p = [ [] for i in range(self.nel) ]

        for i in range(0, numParticles): # For each of the particles
            xP = pCoords[i][:] # particle coordinates
            id = np.zeros(self.dim)
            for j in range(0,self.dim):
                id[j] = np.floor(xP[j] / self.h[j])
                # Catch boundary case
                if(id[j] == self.n[j]):
                    id[j] -= 1
            if(self.dim == 2):
                p2e[i] = self.n[0] * id[1] + id[0]
            else:
                p2e[i] = self.n[0] * self.n[1] * id[1] + self.n[0] * id[1] + id[0]

            self.e2p[p2e[i]].append(i)
            # Global to local mapping is easy as structured grid / domain
            node = self.connectivity[p2e[i]][:]
            for j in range(0,self.dim):
                p_localCoords[i][j] = (2 / self.h[j]) * (pCoords[i][j] - (self.coords[node[0]][j] + 0.5 * self.h[j]))

        self.particle_localCoords = p_localCoords
        self.p2e = p2e

        return p_localCoords, p2e

    def evalPhi(self, x, order = 1):
        if(self.dim == 2):
            if(order == 1):
                phi = np.zeros(4)
                phi[0] = (1 - x[0]) * (1 - x[1])
                phi[1] = (1 + x[0]) * (1 - x[1])
                phi[2] = (1 + x[0]) * (1 + x[1])
                phi[3] = (1 - x[0]) * (1 + x[1])
                phi *= 0.25
        return phi

    def findBoundaryElements(self, boundaryId, overlap):

        elementList = []
        particleList = []

        # BoundaryId Define 4 boundaries of a rectangular domain 0 - LHS the counts anti-clockwise
        for ie in range(0, self.nel): # Loop over all elements

            num_bnd_elem = len(elementList) # Record number of current elements in the list

            nodesInElement = self.connectivity[ie][:]
            if(boundaryId == 0):
                if(self.coords[nodesInElement[1]][0] < self.X0[0] + overlap):
                    elementList.append(ie)
            if(boundaryId == 1):
                if(self.coords[nodesInElement[2]][1] < self.X0[1] + overlap):
                    elementList.append(ie)
            if(boundaryId == 2):
                if(self.coords[nodesInElement[0]][0] > self.X0[0] + self.L[0] - overlap):
                    elementList.append(ie)
            if(boundaryId == 3):
                if(self.coords[nodesInElement[1]][1] > self.X0[1] + self.L[1] - overlap):
                    elementList.append(ie)

            if(num_bnd_elem < len(elementList)): # Element has been added
                # "ie" is the index of the new element added, ad
                particleList.append(self.e2p[ie])

        return elementList, particleList