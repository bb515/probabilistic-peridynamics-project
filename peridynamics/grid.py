import numpy as np

class Grid:
    def __init__(self):
        self.dim = 2
        self.elements = 0
        self.nnnodes = 0

    def  findNeighbours(self):
        # Simple implementation as course mesh will be samll
        M = np.zeros((self.nnodes, self.nnodes), dtype= int)
        
        for ie in range(0, self.elements):
            nodes = self.connectivity[ie][:]
            for i in range(0, 4):
                for j in range(0, 4):
                    if(i != j):
                        M[nodes[i]][nodes[j]] = 1
        # Build List of Neighbours
        self.neighbours = [ [] for i in range(self.nnodes)]
        for i in range(0, self.nnodes):
            for j in range(0, self.nnodes):
                if (M[i][j] == 1):
                    self.neighbours[i].append(int(j))
        # Build list of macroscale elements in which a node lives
        tmp_N2E = [ [] for i in range(self.nnodes)]
        for ie in range(0, self.nel): # for each elements
            nnodes = self.connectivity[ie][:]
            for j in range(0, nnodes.size):
                tmp_N2E[nnodes[j]].append(ie)
        self.node_to_elements = [ [] for i in range(self.nnodes)]
        for i in range(0, self.nnodes):
            tmp_array = np.array(tmp_N2E[i])
            self.node_to_elements[i] = np.unique(tmp_array)
        return self.neighbours, self.node_to_elements

    def build_structured_mesh(self, L, n, X0):
        self.X0 = X0
        self.n = n
        # Function builds a structured finite element mesh in 2D
        nnodes_x = n[0] + 1
        nnodes_y = n[1] + 1
        self.nnodes_per_element = 4
        self.nnodes = nnodes_x * nnodes_y
        self.elements = n[0] * n[1]

        x = np.linspace(X0[0], X0[0] + L[0], nnodes_x)
        y = np.linspace(X0[1], X0[1] + L[1], nnodes_y)

        self.h = np.zeros(self.dim)

        for i in range(0, self.dim):
            self.h[i] = L[i] / n[i]

        # nnodes will be formed from a tensor product of this two vectors
        self.coords = np.zeros((self.nnodes, 2))

        count = 0
        for i in range(0, n[1] + 1):
            for j in range(0, n[0] + 1):
                self.coords[count][0] = x[j]
                self.coords[count][1] = y[i]
                count += 1

        # Build Connectivity Matrix
        self.elements = n[0] * n[1]
        self.connectivity = np.zeros((self.elements, self.nnodes_per_element),
                                     dtype=int)
        count = 0
        ncount = 0
        for j in range(0, n[1]):
            for i in range(0, n[0]):
                self.connectivity[count][0] = ncount
                self.connectivity[count][1] = ncount + 1
                self.connectivity[count][2] = ncount + nnodes_x + 1
                self.connectivity[count][3] = ncount + nnodes_x
                count += 1
                ncount += 1
            ncount += 1

    def particle_to_cell(self, coords_particles):
        particles = int(coords_particles[:].size / self.dim)
        p2e = np.zeros(particles, dtype=int)
        coords_local = np.zeros((particles, self.dim))

        self.e2p = [ [] for i in range(self.elements) ]
        # For each of the particles
        for i in range(0, particles):
            # particle coordinates
            xP = coords_particles[i][:]
            id_ = np.zeros(self.dim)
            for j in range(0, self.dim):
                id_[j] = np.floor(xP[j] / self.h[j])
                # Catch boundary case
                if id_[j] == self.n[j]:
                    id_[j] -= 1
            if self.dim == 2:
                p2e[i] = self.n[0] * id_[1] + id_[0]
            else:
                p2e[i] = (self.n[0] * self.n[1] * id_[1]
                          + self.n[0] * id_[1] + id_[0])
            # Global to local mapping is easy as structured grid / domain
            node = self.connectivity[p2e[i]][:]
            for j in range(0, self.dim):
                coords_local[i][j] = (
                    (2 / self.h[j])
                    * (coords_particles[i][j]
                       - (self.coords[node[0]][j] + 0.5 * self.h[j]))
                    )

        self.coords_local = coords_local
        self.p2e = p2e
        return coords_local, p2e

    def eval_phi(self, x, order = 1):
        """
        Evaluates shape functions at (element?) coordinate x
        : arg x: 2D local element coordinates
        : arg order: int value describing the order of the FEM grid.
        
        : returns phi : the evaluated shape function
        : return type: float
        """
        if self.dim == 2:
            if(order == 1):
                phi = np.zeros(4)
                phi[0] = (1 - x[0]) * (1 - x[1])
                phi[1] = (1 + x[0]) * (1 - x[1])
                phi[2] = (1 + x[0]) * (1 + x[1])
                phi[3] = (1 - x[0]) * (1 + x[1])
                phi *= 0.25
        return phi

    def find_boundary_elements(self, boundary_id, overlap):
        """
        Finds boundary elements
        : arg boundary_id:
        : arg overlap:
        """
        element_list = []
        particle_list = []
        # Boundary_ud define 4 boundaries of a rectuangular domain 0 -
        # LHS the counts anti clockwise
        for i in range(0, self.nel): # loop over all elements
            # Record number of current elements in the list
            num_bnd_element = len(element_list)

            nnodes_in_element = self.connectivity[i][:]
            if boundary_id == 0:
                if self.coords[nnodes_in_element[1]][0] < self.X0[0] + overlap:
                    element_list.append(i)
            if boundary_id == 1:
                if self.coords[nnodes_in_element[2]][1] < self.X0[1] + overlap:
                    element_list.append(i)
            if boundary_id == 2:
                if self.coords[nnodes_in_element[0]][0] > self.X0[0] + self.L[0] - overlap:
                    element_list.append(i)
            if boundary_id == 3:
                if self.coords[nnodes_in_element[1]][1] > self.X0[1] + self.L[1] - overlap:
                    element_list.append(i)
            if num_bnd_element < len(element_list): # Element has been added
                # "ie" is the list index
                particle_list.append(self.e2p[i])

        return element_list, particle_list