"""Peridynamics model."""
from .integrators import Integrator
from collections import namedtuple
import meshio
import numpy as np
import pathlib
from scipy import sparse
from scipy.spatial.distance import cdist
import time
import sys
from .post_processing import vtk

_MeshElements = namedtuple("MeshElements", ["connectivity", "boundary"])
_mesh_elements_2d = _MeshElements(connectivity="triangle",
                                  boundary="line")
_mesh_elements_3d = _MeshElements(connectivity="tetra",
                                  boundary="triangle")

class Model:
    """
    A peridynamics model.
    This class allows users to define a peridynamics system from parameters and
    a set of initial conditions (coordinates and connectivity).
    :Example: ::
        >>> from peridynamics import Model
        >>>
        >>> model = Model(
        >>>     mesh_file="./example.msh",
        >>>     horizon=0.1,
        >>>     critical_strain=0.005,
        >>>     elastic_modulus=0.05
        >>>     )
    To define a crack in the inital configuration, you may supply a list of
    pairs of particles between which the crack is.
    :Example: ::
        >>> from peridynamics import Model, initial_crack_helper
        >>>
        >>> initial_crack = [(1,2), (5,7), (3,9)]
        >>> model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
        >>>               elastic_modulus=0.05, initial_crack=initial_crack)
    If it is more convenient to define the crack as a function you may also
    pass a function to the constructor which takes the array of coordinates as
    its only argument and returns a list of tuples as described above. The
    :func:`peridynamics.model.initial_crack_helper` decorator has been provided
    to easily create a function of the correct form from one which tests a
    single pair of node coordinates and returns `True` or `False`.
    :Example: ::
        >>> from peridynamics import Model, initial_crack_helper
        >>>
        >>> @initial_crack_helper
        >>> def initial_crack(x, y):
        >>>     ...
        >>>     if crack:
        >>>         return True
        >>>     else:
        >>>         return False
        >>>
        >>> model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
        >>>               elastic_modulus=0.05, initial_crack=initial_crack)
    The :meth:`Model.simulate` method can be used to conduct a peridynamics
    simulation. For this an :class:`peridynamics.integrators.Integrator` is
    required, and optionally a function implementing the boundary conditions.
    :Example: ::
        >>> from peridynamics import Model, initial_crack_helper
        >>> from peridynamics.integrators import Euler
        >>>
        >>> model = Model(...)
        >>>
        >>> euler = Euler(dt=1e-3)
        >>>
        >>> indices = np.arange(model.nnodes)
        >>> model.lhs = indices[model.coords[:, 0] < 1.5*model.horizon]
        >>> model.rhs = indices[model.coords[:, 0] > 1.0 - 1.5*model.horizon]
        >>>
        >>> def boundary_function(model, u, step):
        >>>     u[model.lhs] = 0
        >>>     u[model.rhs] = 0
        >>>     u[model.lhs, 0] = -1.0 * step
        >>>     u[model.rhs, 0] = 1.0 * step
        >>>
        >>>     return u
        >>>
        >>> u, damage, *_ = model.simulate(steps=1000, integrator=euler,
        >>>                                boundary_function=boundary_function)
    """

    def __init__(self, mesh_file, horizon, critical_strain, elastic_modulus,
                 initial_crack=[], dimensions=2):
        """
        Construct a :class:`Model` object.
        :arg str mesh_file: Path of the mesh file defining the systems nodes
            and connectivity.
        :arg float horizon: The horizon radius. Nodes within `horizon` of
            another interact with that node and are said to be within its
            neighbourhood.
        :arg float critical_strain: The critical strain of the model. Bonds
            which exceed this strain are permanently broken.
        :arg float elastic_modulus: The appropriate elastic modulus of the
            material.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack. Default is []
        :type initial_crack: list(tuple(int, int)) or function
        :arg int dimensions: The dimensionality of the model. The
            default is 2.
        :returns: A new :class:`Model` object.
        :rtype: Model
        :raises DimensionalityError: when an invalid `dimensions` argument is
            provided.
        """
        # Is the mesh transfinite mesh (support regular grid spacing with
        #cuboidal (not tetra) elements (default 0))
        self.transfinite = 0
        # Are the stiffness correction factors calculated using mesh
        # element volumes (default 'precise', 1) or average nodal 
        # volume of a transfinite mesh (0)  
        #self.precise_stiffness_correction = 1
        # Set model dimensionality
        self.dimensions = dimensions

        if dimensions == 2:
            self.mesh_elements = _mesh_elements_2d
        elif dimensions == 3:
            self.mesh_elements = _mesh_elements_3d
        else:
            raise DimensionalityError(dimensions)

        # Read coordinates and connectivity from mesh file
        self._read_mesh(mesh_file)

        self.horizon = horizon
        self.critical_strain = critical_strain

        # Determine bond stiffness
        self.bond_stiffness = (
            18.0 * elastic_modulus / (np.pi * self.horizon**4)
            )

        # Read coordinates and connectivity from mesh file
        self._read_mesh(mesh_file)

        # Calculate the volume for each node
        self.volume = self._volume()

        # Determine neighbours
        neighbourhood = self._neighbourhood()

        # Set family, the number of neighbours for each node
        self.family = np.squeeze(np.array(np.sum(neighbourhood, axis=0)))

        # Set the initial connectivity
        self.initial_connectivity = self._connectivity(neighbourhood,
                                                       initial_crack)

        # Set the node distance and failure strain matrices
        _, _, _, self.L_0 = self._H_and_L(self.coords,
                                          self.initial_connectivity)

    def _read_mesh(self, filename):
        """
        Read the model's nodes, connectivity and boundary from a mesh file.
        :arg str filename: Path of the mesh file to read
        :returns: None
        :rtype: NoneType
        """
        mesh = meshio.read(filename)

        if self.transfinite == 1:
            # In this case, only need coordinates, encoded as mesh points
            self.coords = mesh.points
            self.nnodes = self.coords.shape[0]

        else:
            # Get coordinates, encoded as mesh points
            self.coords = mesh.points
            self.nnodes = self.coords.shape[0]

            # Get connectivity, mesh triangle cells
            self.mesh_connectivity = mesh.cells[self.mesh_elements.connectivity]

            # Get boundary connectivity, mesh lines
            self.mesh_boundary = mesh.cells[self.mesh_elements.boundary]

            # Get number elements on boundary?
            self.nelem_bnd = self.mesh_boundary.shape[0]
    def write_mesh(self, filename, damage=None, displacements=None,
                   file_format=None):
        """
        Write the model's nodes, connectivity and boundary to a mesh file.
        :arg str filename: Path of the file to write the mesh to.
        :arg damage: The damage of each node. Default is None.
        :type damage: :class:`numpy.ndarray`
        :arg displacements: An array with shape (nnodes, dim) where each row is
            the displacement of a node. Default is None.
        :type displacements: :class:`numpy.ndarray`
        :arg str file_format: The file format of the mesh file to
            write. Inferred from `filename` if None. Default is None.
        :returns: None
        :rtype: NoneType
        """
        meshio.write_points_cells(
            filename,
            points=self.coords,
            cells=[
                (self.mesh_elements.connectivity, self.mesh_connectivity),
                (self.mesh_elements.boundary, self.mesh_boundary)
                ],
            point_data={
                "damage": damage,
                "displacements": displacements
                },
            file_format=file_format
            )

    def _volume(self):
        """
        Calculate the value of each node.
        :returns: None
        :rtype: NoneType
        """
        volume = np.zeros(self.nnodes)

        for element in self.mesh_connectivity:
            # Compute area / volume
            val = 1. / len(element)

            # Define area of element
            if (self.dimensions == 2):
                xi, yi, *_ = self.coords[element[0]]
                xj, yj, *_ = self.coords[element[1]]
                xk, yk, *_ = self.coords[element[2]]
                val *= 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

            volume[element] += val

        return volume

    def _neighbourhood(self):
        """
        Determine the neighbourhood of all nodes.
        :returns: The sparse neighbourhood matrix.  Element [i, j] of this
            martrix is True if i is within `horizon` of j and False otherwise.
        :rtype: :class:`scipy.sparse.csr_matrix`
        """
        # Calculate the Euclidean distance between each pair of nodes
        distance = cdist(self.coords, self.coords, 'euclidean')

        # Construct the neighbourhood matrix (neighbourhood[i, j] = True if i
        # and j are neighbours)
        nnodes = self.nnodes
        neighbourhood = np.zeros((nnodes, nnodes), dtype=np.bool)
        # Connect nodes which are within horizon of each other
        neighbourhood[distance < self.horizon] = True

        return sparse.csr_matrix(neighbourhood)

    def _connectivity(self, neighbourhood, initial_crack):
        """
        Initialise the connectivity.
        :arg neighbourhood: The sparse neighbourhood matrix.
        :type neighbourhood: :class:`scipy.sparse.csr_matrix`
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack.
        :type initial_crack: list(tuple(int, int)) or function
        :returns: The sparse connectivity matrix. Element [i, j] of this matrix
            is True if i and j are bonded and False otherwise.
        :rtype: :class:`scipy.sparse.csr_matrix`
        """
        if callable(initial_crack):
            initial_crack = initial_crack(self.coords, neighbourhood)

        # Construct the initial connectivity matrix
        conn = neighbourhood.toarray()
        for i, j in initial_crack:
            # Connectivity is symmetric
            conn[i, j] = False
            conn[j, i] = False
        # Nodes are not connected with themselves
        np.fill_diagonal(conn, False)

        # Lower triangular - count bonds only once
        # make diagonal values False
        conn = np.tril(conn, -1)

        # Convert to sparse matrix
        return sparse.csr_matrix(conn)

    @staticmethod
    def _displacements(r):
        """
        Determine displacements, in each dimension, between vectors.
        :arg r: A (n,3) array of coordinates.
        :type r: :class:`numpy.ndarray`
        :returns: A tuple of three arrays giving the displacements between
            each pair of parities in the first, second and third dimensions
            respectively. m[i, j] is the distance from j to i (i.e. i - j).
        :rtype: tuple(:class:`numpy.ndarray`)
        """
        n = len(r)
        x = np.tile(r[:, 0], (n, 1))
        y = np.tile(r[:, 1], (n, 1))
        z = np.tile(r[:, 2], (n, 1))

        d_x = x.T - x
        d_y = y.T - y
        d_z = z.T - z

        return d_x, d_y, d_z

    def _H_and_L(self, r, connectivity):
        """
        Construct the displacement and distance matrices.
        The H matrices  are sparse matrices containing displacements in
        a particular dimension and the L matrix isa sparse matrix containing
        the Euclidean distance. Elements for particles which are not connected
        are 0.
        :arg r: The positions of all nodes.
        :type r: :class:`numpy.ndarray`
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`
        :returns: (H_x, H_y, H_z, L) A tuple of sparse matrix. H_x, H_y and H_z
            are the matrices of displacements between pairs of particles in the
            x, y and z dimensions respectively. L is the Euclidean distance
            between pairs of particles.
        :rtype: tuple(:class:`scipy.sparse.csr_matrix`)
        """
        # Get displacements in each dimension between coordinate
        H_x, H_y, H_z = self._displacements(r)

        # Convert to spare matrices filtered by the connectivity matrix (i.e.
        # only for particles which interact).
        a = connectivity + connectivity.transpose()
        H_x = a.multiply(H_x)
        H_y = a.multiply(H_y)
        H_z = a.multiply(H_z)

        L = (H_x.power(2) + H_y.power(2) + H_z.power(2)).sqrt()

        return H_x, H_y, H_z, L

    def _strain(self, u, L):
        """
        Calculate the strain of all bonds for a given displacement.
        :arg u: The displacement array with shape (`nnodes`, `dimension`).
        :type u: :class:`numpy.ndarray`
        :arg L: The euclidean distance between each pair of nodes.
        :type L: :class:`scipy.sparse.csr_matrix`
        :returns: The strain between each pair of nodes.
        :rtype: :class:`scipy.sparse.lil_matrix`
        """
        # Calculate difference in bond lengths from the initial state
        dL = L - self.L_0

        # Calculate strain
        nnodes = self.nnodes
        strain = sparse.lil_matrix((nnodes, nnodes))
        non_zero = self.L_0.nonzero()
        strain[non_zero] = (dL[non_zero]/self.L_0[non_zero])

        return strain

    def _break_bonds(self, strain, connectivity):
        """
        Break bonds which have exceeded the critical strain.
        :arg strain: The strain of each bond.
        :type strain: :class:`scipy.sparse.lil_matrix`
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`
        :returns: The updated connectivity.
        :rtype: :class:`scipy.sparse.csr_matrix`
        """
        unbroken = sparse.lil_matrix(connectivity.shape)

        # Find broken bonds
        nnodes = self.nnodes
        critical_strains = np.full((nnodes, nnodes), self.critical_strain)
        connected = connectivity.nonzero()
        unbroken[connected] = (
            critical_strains[connected] - abs(strain[connected])
            ) > 0

        connectivity = sparse.csr_matrix(unbroken)

        return connectivity

    def _damage(self, connectivity):
        """
        Calculate bond damage.
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`
        :returns: A (`nnodes`, ) array containing the damage for each node.
        :rtype: :class:`numpy.ndarray`
        """
        family = self.family
        # Sum all unbroken bonds for each node
        unbroken_bonds = (connectivity + connectivity.transpose()).sum(axis=0)
        # Convert matrix object to array
        unbroken_bonds = np.squeeze(np.array(unbroken_bonds))

        # Calculate damage for each node
        damage = np.divide((family - unbroken_bonds), family)

        return damage

    def _bond_force(self, strain, connectivity, L, H_x, H_y, H_z):
        """
        Calculate the force due to bonds acting on each node.
        :arg strain: The strain of each bond.
        :type strain: :class:`scipy.sparse.lil_matrix`
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`
        :arg L: The Euclidean distance between pairs of nodes.
        :type L: :class:`scipy.sparse.csr_matrix`
        :arg H_x: The displacement in the x dimension between each pair of
            nodes.
        :type H_x: :class:`scipy.sparse.csr_matrix`
        :arg H_y: The displacement in the y dimension between each pair of
            nodes.
        :type H_y: :class:`scipy.sparse.csr_matrix`
        :arg H_z: The displacement in the z dimension between each pair of
            nodes.
        :type H_z: :class:`scipy.sparse.csr_matrix`
        :returns: A (`nnodes`, 3) array of the component of the force in each
            dimension for each node.
        :rtype: :class:`numpy.ndarray`
        """
        # Calculate the normalised forces
        force_normd = sparse.lil_matrix(connectivity.shape)
        connected = connectivity.nonzero()
        force_normd[connected] = strain[connected] / L[connected]

        # Make lower triangular into full matrix
        force_normd.tocsr()
        force_normd = force_normd + force_normd.transpose()

        # Calculate component of force in each dimension
        bond_force_x = force_normd.multiply(H_x)
        bond_force_y = force_normd.multiply(H_y)
        bond_force_z = force_normd.multiply(H_z)

        # Calculate total force on nodes in each dimension
        F_x = np.squeeze(np.array(bond_force_x.sum(axis=0)))
        F_y = np.squeeze(np.array(bond_force_y.sum(axis=0)))
        F_z = np.squeeze(np.array(bond_force_z.sum(axis=0)))

        # Determine actual force
        F = np.stack((F_x, F_y, F_z), axis=-1)
        F *= self.volume.reshape((self.nnodes, 1))
        F *= self.bond_stiffness

        return F

    def simulate(self, steps, integrator, boundary_function=None, u=None,
                 connectivity=None, first_step=1, write=None, write_path=None):
        """
        Simulate the peridynamics model.
        :arg int steps: The number of simulation steps to conduct.
        :arg  integrator: The integrator to use, see
            :mod:`peridynamics.integrators` for options.
        :type integrator: :class:`peridynamics.integrators.Integrator`
        :arg boundary_function: A function to apply the boundary conditions for
            the simlation. It has the form
            boundary_function(:class:`peridynamics.model.Model`,
            :class:`numpy.ndarray`, `int`). The arguments are the model being
            simulated, the current displacements, and the current step number
            (beginning from 1). `boundary_function` returns a (nnodes, 3)
            :class:`numpy.ndarray` of the updated displacements
            after applying the boundary conditions. Default `None`.
        :type boundary_function: function
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used.
        :type connectivity: :class:`scipy.sparse.csr_matrix` or
            :class:`numpy.ndarray`
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation, especially if `boundary_function` depends
            on the absolute step number.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If `None` then
            no output is written. Default `None`.
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str
        :returns: A tuple of the final displacements (`u`), damage and
            connectivity.
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`scipy.sparse.csr_matrix`)
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)

        # Create initial displacements is none is provided
        if u is None:
            u = np.zeros((self.nnodes, 3))

        # Use the initial connectivity (when the Model was constructed) if none
        # is provided
        if connectivity is None:
            connectivity = self.initial_connectivity
        elif type(connectivity) == np.ndarray:
            connectivity = sparse.csr_matrix(connectivity)

        # Create dummy boundary conditions function is none is provided
        if boundary_function is None:
            def boundary_function(model, u, step):
                return u

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)

        for step in range(first_step, first_step+steps):
            # Get current distance between nodes (i.e. accounting for
            # displacements)
            H_x, H_y, H_z, L = self._H_and_L(self.coords+u, connectivity)

            # Calculate the strain of each bond
            strain = self._strain(u, L)

            # Update the connectivity and calculate the current damage
            connectivity = self._break_bonds(strain, connectivity)
            damage = self._damage(connectivity)

            # Calculate the bond due to forces on each node
            f = self._bond_force(strain, connectivity, L, H_x, H_y, H_z)

            # Conduct one integration step
            u = integrator(u, f)

            # Apply boundary conditions
            u = boundary_function(self, u, step)

            if write:
                if step % write == 0:
                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

        return u, damage, connectivity

class OpenCL(Model):
    """
    A peridynamics model using OpenCL.
    This class allows users to define a peridynamics system from parameters and
    a set of initial conditions (coordinates and connectivity).
    """
    def __init__(self, mesh_file_name, 
                 density = None,
                 horizon = None, 
                 damping = None,
                 dx = None,
                 bond_stiffness_concrete = None,
                 bond_stiffness_steel = None, 
                 critical_strain_concrete = None,
                 critical_strain_steel = None,
                 crack_length = None,
                 volume_total=None,
                 bond_type=None,
                 network_file_name = 'Network.vtk',
                 initial_crack=[],
                 dimensions=2,
                 transfinite= None,
                 precise_stiffness_correction = None):
        """
        Construct a :class:`OpenCL` object, which inherits Model class.
        
        :arg str mesh_file_name: Path of the mesh file defining the systems nodes
            and connectivity.
        :arg float density: Density of the bulk material in kg/m^3
        :arg float horizon: The horizon radius. Nodes within `horizon` of
            another interact with that node and are said to be within its
            neighbourhood.
        :arg float family_volume: The spherical volume defined by the horizon
        radius.
        :arg float critical_strain: The critical strain of the model. Bonds
            which exceed this strain are permanently broken.
        :arg float elastic_modulus: The appropriate elastic modulus of the
            material.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack. Default is []
        :type initial_crack: list(tuple(int, int)) or function
        :arg int dimensions: The dimensionality of the model. The
            default is 2.
        :returns: A new :class:`Model` object.
        :rtype: Model
        :raises DimensionalityError: when an invalid `dimensions` argument is
            provided.
        """

        # verbose
        self.v = False

        if dimensions == 2:
            self.mesh_elements = _mesh_elements_2d
            self.family_volume = np.pi*np.power(horizon, 2) * 0.001
        elif dimensions == 3:
            self.mesh_elements = _mesh_elements_3d
            self.family_volume = (4./3)*np.pi*np.power(horizon, 3)
        else:
            raise DimensionalityError(dimensions)

        self.dimensions = dimensions
        self.degrees_freedom = 3

        # bb515 Are the stiffness correction factors calculated using mesh element
        # volumes (default 'precise', 1) or average nodal volume of a transfinite
        # mesh (0)      
        self.precise_stiffness_correction = precise_stiffness_correction
        # bb515 Is the mesh transfinite mesh (support regular grid spacing with 
        # cuboidal (not tetra) elements, look up "gmsh transfinite") (default 0)
        # I'm only planning on using this for validation against literature
        self.transfinite = transfinite
        # Peridynamics parameters. These parameters will be passed to openCL
        # kernels by command line argument Bond-based peridynamics, known in
        # PDLAMMPS as Prototype Microelastic Brittle (PMB) Model requires a
        #poisson ratio of v = 0.25, but this makes little to no difference 
        # in quasi-brittle materials
        self.poisson_ratio = 0.25
        # These are the parameters that the user needs to define
        self.density = density
        self.horizon = horizon
        self.damping = damping
        self.dx = dx
        self.bond_stiffness_concrete = bond_stiffness_concrete
        self.bond_stiffness_steel = bond_stiffness_steel
        self.critical_strain_concrete = critical_strain_concrete
        self.critical_strain_steel = critical_strain_steel
        self.crack_length = crack_length
        self.volume_total = volume_total
        self.network_file_name = network_file_name
        self.dt = None
        self.max_reaction = None
        self.load_scale_rate = None

        self._read_mesh(mesh_file_name)

        st = time.time()

        self._set_volume(volume_total)

        # If the network has already been written to file, then read, if not, setNetwork
        try:
            self._read_network(self.network_file_name)
        except:
            print('No network file found: writing network file.')
            self._set_network(self.horizon, bond_type)

        # Initate crack
        self._set_connectivity(initial_crack)

        if self.v == True:
            print(
                "Building horizons took {} seconds. Horizon length: {}".format(
                    (time.time() - st), self.max_horizon_length))

        # Initiate boundary condition containers
        self.bc_types = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.intc)
        self.bc_values = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.float64)
        self.tip_types = np.zeros(self.nnodes, dtype=np.intc)

        if self.v == True:
            print("sum total volume", self.sum_total_volume)
            print("user input volume total", volume_total)

    def _read_network(self, network_file):
        """ For reading a network file if it has been written to file yet.
        Significantly quicker than building horizons from scratch, however
        the network file size is quite large for large node num.
        :arg network_file: the network vtk file including information about
        node families, bond stiffnesses, critical stretches, number of nodes
        , max horizon length and horizon lengths.
        """
        def find_string(string, iline):
            """
            Function for incrimenting the line when reading the vtk file,
            network_file until input string is found
            :
            :arg string: The string in the vtk to be found.
            :arg iline: The current count of the line no. in the read of
            'network_file'
            :returns: list of strings of row of the chosen line
            :rtype: list
            """
            found = 0
            while (found == 0):
                iline+= 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                found = 1 if string in row_as_list else 0
            return row_as_list, iline
        
        f = open(network_file, "r")

        if f.mode == "r":
            iline = 0

            # Read the Max horizons length first
            row_as_list, iline = find_string('MAX_HORIZON_LENGTH', iline)
            max_horizon_length = int(row_as_list[1])
            if self.v == True:
                print('max_horizon_length', max_horizon_length)
            # Read nnodes
            row_as_list, iline = find_string('NNODES', iline)
            nnodes = int(row_as_list[1])
            if self.v == True:
                print('nnodes', nnodes)
            # Read horizons lengths
            row_as_list, iline = find_string('HORIZONS_LENGTHS', iline)
            horizons_lengths = np.zeros(nnodes, dtype=int)
            for i in range(0, nnodes):
                iline += 1
                line = f.readline()
                horizons_lengths[i] = np.intc(line.split())

            # Read family matrix
            if self.v == True:
                print('Building family matrix from file')
            row_as_list, iline = find_string('FAMILY', iline)
            family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                family.append(np.zeros(len(row_as_list), dtype=np.intc))
                for j in range(0, len(row_as_list)):
                    family[i][j] = np.intc(row_as_list[j])

            # Read stiffness values
            if self.v == True:
                print('Building stiffnesses from file')
            row_as_list, iline = find_string('STIFFNESS', iline)
            bond_stiffness_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                bond_stiffness_family.append(np.zeros(len(row_as_list), dtype=np.float64))
                for j in range(0, len(row_as_list)):
                    bond_stiffness_family[i][j] = (row_as_list[j])

            # Now read critcal stretch values
            if self.v == True:
                print('Building critical stretch values from file')
            row_as_list, iline = find_string('STRETCH', iline)
            bond_critical_stretch_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                bond_critical_stretch_family.append(np.zeros(len(row_as_list), dtype=np.float64))
                for j in range(0, len(row_as_list)):
                    bond_critical_stretch_family[i][j] = row_as_list[j]

            # Maximum number of nodes that any one of the nodes is connected to
            max_horizon_length_check = np.intc(
                    1<<(len(max(family, key=lambda x: len(x)))-1).bit_length()
                )
            assert max_horizon_length == max_horizon_length_check, 'Read failed on MAX_HORIZON_LENGTH check'

            horizons = -1 * np.ones([nnodes, max_horizon_length])
            for i, j in enumerate(family):
                horizons[i][0:len(j)] = j

            bond_stiffness = -1. * np.ones([nnodes, max_horizon_length])
            for i, j in enumerate(bond_stiffness_family):
                bond_stiffness[i][0:len(j)] = j

            bond_critical_stretch = -1. * np.ones([nnodes, max_horizon_length])
            for i, j in enumerate(bond_critical_stretch_family):
                bond_critical_stretch[i][0:len(j)] = j

            # Make sure it is in a datatype that C can handle
            self.horizons = horizons.astype(np.intc)
            self.bond_stiffness = bond_stiffness
            self.bond_critical_stretch = bond_critical_stretch
            self.horizons_lengths = horizons_lengths.astype(np.intc)
            self.family = family
            self.max_horizon_length = max_horizon_length
            self.nnodes = nnodes
            f.close()

    def _set_volume(self, volume_total):
        """
        Calculate the value of each node.
        
        :arg volume_total: User input for the total volume of the mesh, for checking sum total of elemental volumes is equal to user input volume for simple prismatic problems.
        In the case of non-prismatic problems when the user does not know what the volume is, we should do something else as an assertion
        :returns: None
        :rtype: NoneType
        """
        # bb515 this has changed significantly from the sequential code.
        # OpenCL (or rather C) requires that we are careful with
        # types so that they are compatible with the specifed C types in the
        # OpenCL kernels
        self.V = np.zeros(self.nnodes, dtype=np.float64)

        # this is the sum total of the elemental volumes, initiated at 0.
        self.sum_total_volume = 0

        if self.transfinite == 1:
            """ Tranfinite mode is when we have approximated the volumes of the nodes
            as the average volume of nodes on a rectangular grid.
            The transfinite grid (search on youtube for "transfinite mesh gmsh") is not
            neccessarily made up of tetrahedra, but may be made up of cuboids.
            """
            tmp = volume_total / self.nnodes
            for i in range(0, self.nnodes):
                self.V[i] = tmp
                self.sum_total_volume += tmp
        else:
            for element in self.mesh_connectivity:

                # Compute Area or Volume
                val = 1. / len(element)

                # Define area of element
                if self.dimensions == 2:

                    xi, yi, *_ = self.coords[element[0]]
                    xj, yj, *_ = self.coords[element[1]]
                    xk, yk, *_ = self.coords[element[2]]

                    element_area = (
                            0.5 * np.absolute((
                                    (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)))
                            )
                    val *= element_area
                    self.sum_total_volume += element_area

                elif self.dimensions == 3:

                    a = self.coords[element[0]]
                    b = self.coords[element[1]]
                    c = self.coords[element[2]]
                    d = self.coords[element[3]]

                    # Volume of a tetrahedron
                    i = np.subtract(a,d)
                    j = np.subtract(b,d)
                    k = np.subtract(c,d)

                    element_volume = (1./6) * np.absolute(np.dot(i, np.cross(j,k)))
                    val*= element_volume
                    self.sum_total_volume += element_volume
                else:
                    raise ValueError('dim', 'dimension size can only take values 2 or 3')

                for j in range(0, len(element)):
                    self.V[element[j]] += val

        # For non prismatic problems where the user does not know the 
        # volume_total, do another test?
        assert self.sum_total_volume - volume_total < volume_total/1e5, \
        "Total volume not as expected: total of elemental volumes was {},\
        but expected total volume was {}".format(self.sum_total_volume, volume_total)
        self.V = self.V.astype(np.float64)

    def _set_network(self, horizon, bond_type):
        """
        Sets the family matrix, and converts this to a horizons matrix 
        (a fixed size data structure compatible with OpenCL).
        Calculates horizons_lengths
        Also initiate crack here if there is one
        :arg horizon: Peridynamic horizon distance
        :returns: None
        :rtype: NoneType
        """
        def l2(y1, y2):
            """
            Euclidean distance between nodes y1 and y2.
            """
            l2 = 0
            for i in range(len(y1)):
                l2 += (y1[i] - y2[i]) * (y1[i] - y2[i])
            l2 = np.sqrt(l2)
            return l2

        # Container for nodal family
        family = []
        bond_stiffness_family = []
        bond_critical_stretch_family = []

        # Container for number of nodes (including self) that each of the nodes
        # is connected to
        self.horizons_lengths = np.zeros(self.nnodes, dtype=np.intc)

        for i in range(0, self.nnodes):
            print('node', i, 'networking...')
            # Container for family nodes
            tmp = []
            # Container for bond stiffnesses
            tmp2 = []
            # Container for bond critical stretches
            tmp3 = []
            for j in range(0, self.nnodes):
                if i != j:
                    distance = l2(self.coords[i, :], self.coords[j, :])
                    if distance < horizon:
                        tmp.append(j)
                        # Determine the material properties for that bond
                        material_flag = bond_type(self.coords[i, :], self.coords[j, :])
                        if material_flag == 'steel':
                            tmp2.append(self.bond_stiffness_steel)
                            tmp3.append(self.critical_strain_steel)
                        elif material_flag == 'interface':
                            tmp2.append(self.bond_stiffness_concrete * 3.0) # factor of 3 is used for interface bonds in the literature turn this off for parameter est. tests
                            tmp3.append(self.critical_strain_concrete * 3.0) # 3.0 is used for interface bonds in the literature
                        elif material_flag == 'concrete':
                            tmp2.append(self.bond_stiffness_concrete)
                            tmp3.append(self.critical_strain_concrete)

            family.append(np.zeros(len(tmp), dtype=np.intc))
            bond_stiffness_family.append(np.zeros(len(tmp2), dtype=np.float64))
            bond_critical_stretch_family.append(np.zeros(len(tmp3), dtype=np.float64))

            self.horizons_lengths[i] = np.intc((len(tmp)))
            for j in range(0, len(tmp)):
                family[i][j] = np.intc(tmp[j])
                bond_stiffness_family[i][j] = np.float64(tmp2[j])
                bond_critical_stretch_family[i][j] = np.float64(tmp3[j])

        assert len(family) == self.nnodes
        # As numpy array
        self.family = np.array(family)

        # Do the bond critical ste
        self.bond_critical_stretch_family = np.array(bond_critical_stretch_family)
        self.bond_stiffness_family = np.array(bond_stiffness_family)

        self.family_v = np.zeros(self.nnodes)
        for i in range(0, self.nnodes):
            tmp = 0 # tmp family volume
            family_list = family[i]
            for j in range(0, len(family_list)):
                tmp += self.V[family_list[j]]
            self.family_v[i] = tmp

        if self.precise_stiffness_correction == 1:
            # Calculate stiffening factor nore accurately using actual nodal volumes
            for i in range(0, self.nnodes):
                family_list = family[i]
                nodei_family_volume = self.family_v[i]
                for j in range(len(family_list)):
                    nodej_family_volume = self.family_v[j]
                    stiffening_factor = 2.* self.family_volume /  (nodej_family_volume + nodei_family_volume)
                    print('Stiffening factor {}'.format(stiffening_factor))
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 0:
            # TODO: check this code, it was 23:52pm
            average_node_volume = self.volume_total/self.nnodes
            # Calculate stiffening factor - surface corrections for 2D/3D problem, for this we need family matrix
            for i in range(0, self.nnodes):
                nnodes_i_family = len(family[i])
                nodei_family_volume = nnodes_i_family * average_node_volume # Possible to calculate more exactly, we have the volumes for free
                for j in range(len(family[i])):
                    nnodes_j_family = len(family[j])
                    nodej_family_volume = nnodes_j_family* average_node_volume # Possible to calculate more exactly, we have the volumes for free
                    
                    stiffening_factor = 2.* self.family_volume /  (nodej_family_volume + nodei_family_volume)
                    
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 2:
            # Don't apply stiffness correction factor
            pass

        # Maximum number of nodes that any one of the nodes is connected to, must be a power of 2 (for OpenCL reduction)
        self.max_horizon_length = np.intc(
                    1<<(len(max(family, key=lambda x: len(x)))-1).bit_length()
                )

        self.horizons = -1 * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.family):
            self.horizons[i][0:len(j)] = j

        self.bond_stiffness = -1. * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.bond_stiffness_family):
            self.bond_stiffness[i][0:len(j)] = j

        self.bond_critical_stretch = -1. * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.bond_critical_stretch_family):
            self.bond_critical_stretch[i][0:len(j)] = j

        # Make sure it is in a datatype that C can handle
        self.horizons = self.horizons.astype(np.intc)

        vtk.writeNetwork(self.network_file_name, "Network",
                      self.max_horizon_length, self.horizons_lengths,
                      self.family, self.bond_stiffness_family, self.bond_critical_stretch_family)

    def _set_connectivity(self, initial_crack):
        """
        Sets the intial crack.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack.
        :type initial_crack: list(tuple(int, int)) or function
        :returns: None
        :rtype: NoneType
        
        
        bb515 connectivity matrix is replaced by self.horizons and self.horizons_lengths for OpenCL
        
        also see self.family, which is a verlet list:
            self.horizons and self.horizons_lengths are neccessary OpenCL cannot deal with non fixed length arrays
        """
        if self.v == True:
            print("defining crack")
        # This code is the fastest because it doesn't have to iterate through
        # all possible initial crack bonds.
        def is_crack(x, y):
            output = 0
            crack_length = 0.3
            p1 = x
            p2 = y
            if x[0] > y[0]:
                p2 = x
                p1 = y
            # 1e-6 makes it fall one side of central line of particles
            if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
                # draw a straight line between them
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                c = p1[1] - m * p1[0]
                # height a x = 0.5
                height = m * 0.5 + c
                if (height > 0.5 * (1 - crack_length)
                        and height < 0.5 * (1 + crack_length)):
                    output = 1
            return output

        for i in range(0, self.nnodes):
            for k in range(0, self.max_horizon_length):
                j = self.horizons[i][k]
                if is_crack(self.coords[i, :], self.coords[j, :]):
                    self.horizons[i][k] = np.intc(-1)
# =============================================================================
#         # bb515 this code is really slow due to the non symmetry of
#         # self.horizons
#         if callable(initial_crack):
#             initial_crack = initial_crack(self.coords)
#         for i,j in initial_crack:
#             for k, m in enumerate(self.horizons[i]):
#                 if m == j:
#                     self.horizons[i][k] = -1
#             # connectivity is symmetric
#             for k, m in enumerate(self.horizons[j]):
#                 if m == i:
#                     self.horizons[j][k] = -1
# =============================================================================
# =============================================================================
#         # This does the same thing, but slower.
#         if callable(initial_crack):
#             initial_crack = initial_crack(self.coords)
#         for i, j in initial_crack:
#             for k in range(len(self.horizons[i])):
#                 if self.horizons[i][k] == j:
#                     self.horizons[i][k] = np.intc(-1)
#                 # Connectivity is symmetric
#                 if self.horizons[j][k] == i:
#                     self.horizons[j][k] = np.intc(-1)
# =============================================================================
    def _set_D(self, bond_stiffness_const, critical_stretch_const):
        """
        Constructs the failure strains matrix and H matrix, which is a sparse
        matrix containing distances.
        :returns: None
        :rtype: NoneType
        """
        # Set model parameters
        self.bond_stiffness_const = bond_stiffness_const
        self.critical_stretch_const = critical_stretch_const

    def simulate(self, model, sample, steps, integrator, write=None, toolbar=0,
                 displacement_rate = None,
                 build_displacement = None,
                 final_displacement = None):
        """
        Simulate the peridynamics model.
        :arg int steps: The number of simulation steps to conduct.
        :arg  integrator: The integrator to use, see
            :mod:`peridynamics.integrators` for options.
        :type integrator: :class:`peridynamics.integrators.Integrator`
        :arg boundary_function: A function to apply the boundary conditions for
            the simlation. It has the form
            boundary_function(:class:`peridynamics.model.Model`,
            :class:`numpy.ndarray`, `int`). The arguments are the model being
            simulated, the current displacements, and the current step number
            (beginning from 1). `boundary_function` returns a (nnodes, 3)
            :class:`numpy.ndarray` of the updated displacements
            after applying the boundary conditions. Default `None`.
        :type boundary_function: function
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling
            :meth:`peridynamics.model.Model.write_mesh`. If `None` then no
            output is written. Default `None`.
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)
        # Calculate number of time steps that displacement load is in the 'build-up' phase
        if not ((displacement_rate is None) or (build_displacement is None) or (final_displacement is None)):
            build_time, a, b, c= _calc_build_time(build_displacement, displacement_rate, steps)

        # Container for plotting data
        damage_sum_data = []
        tip_displacement_data = []
        tip_force_data = []

        #Progress bar
        toolbar_width = 40
        # Ease off displacement loading switch
        ease_off = 0
        if toolbar:    
            sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        st = time.time()
        for step in range(1, steps+1):
            # Conduct one integration step
            integrator.runtime(model)
            if write:
                if step % write == 0:
                    ft = time.time()
                    damage_data, tip_displacement, tip_force = integrator.write(model, step, sample)
                    tip_displacement_data.append(tip_displacement)
                    tip_force_data.append(tip_force)
                    damage_sum = np.sum(damage_data)
                    damage_sum_data.append(damage_sum)
                    if damage_sum > 0.02*model.nnodes:
                        print('Warning: over 2% of bonds have broken! -- PERIDYNAMICS SIMULATION CONTINUING')
                    elif damage_sum > 0.7*model.nnodes:
                        print('Warning: over 7% of bonds have broken! -- PERIDYNAMICS SIMULATION STOPPING')
                        break
                    if toolbar == 0:
                        print('Print number {}/{} complete in {} s '.format(int(step/write), int(steps/write), time.time() - st))
                        print('Print number {}/{} runtime was ~ {} s '.format(int(step/write), int(steps/write), ft - st))
                        st = time.time()

            # Increase load in linear increments
            if not (model.load_scale_rate is None):
                load_scale = min(1.0, model.load_scale_rate * step)
                if load_scale != 1.0:
                    integrator.incrementLoad(model, load_scale)
            # Increase dispalcement in 5th order polynomial increments
            if not ((displacement_rate is None) or (build_displacement is None) or (final_displacement is None)):
                # 5th order polynomial/ linear curve used to calculate displacement_scale
                displacement_scale, ease_off = _calc_load_displacement_rate(a, b, c,
                                                                 final_displacement,
                                                                 build_time,
                                                                 displacement_rate,
                                                                 step, 
                                                                 build_displacement,
                                                                 ease_off)
                if displacement_scale != 0.0:
                    integrator.incrementDisplacement(model, displacement_scale)
            # No user specified build up parameters case
            elif not (displacement_rate is None):
                integrator.incrementDisplacement(model, 1.0)
            # Loading bar update
            if step%(steps/toolbar_width)<1 & toolbar:
                sys.stdout.write("\u2588")
                sys.stdout.flush()

        if toolbar:
            sys.stdout.write("]\n")

        return damage_sum_data, tip_displacement_data, tip_force
class OpenCLProbabilistic(OpenCL):
    """
    A peridynamics model using OpenCL.
    This class allows users to define a peridynamics system from parameters and
    a set of initial conditions (coordinates and connectivity).
    """
    def __init__(self, mesh_file_name, 
                 density = None,
                 horizon = None, 
                 damping = None,
                 dx = None,
                 bond_stiffness_const = None,
                 critical_stretch_const = None,
                 sigma = None, 
                 l = None,
                 crack_length = None,
                 volume_total=None,
                 bond_type=None,
                 network_file_name = 'Network.vtk',
                 initial_crack=[],
                 dimensions=3,
                 transfinite= None,
                 precise_stiffness_correction = None):
        """
        Construct a :class:`OpenCL` object, which inherits Model class.
        
        :arg str mesh_file_name: Path of the mesh file defining the systems nodes
            and connectivity.
        :arg float density: Density of the bulk material in kg/m^3
        :arg float horizon: The horizon radius. Nodes within `horizon` of
            another interact with that node and are said to be within its
            neighbourhood.
        :arg float family_volume: The spherical volume defined by the horizon
        radius.
        :arg float critical_strain: The critical strain of the model. Bonds
            which exceed this strain are permanently broken.
        :arg float elastic_modulus: The appropriate elastic modulus of the
            material.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack. Default is []
        :type initial_crack: list(tuple(int, int)) or function
        :arg int dimensions: The dimensionality of the model. The
            default is 2.
        :returns: A new :class:`Model` object.
        :rtype: Model
        :raises DimensionalityError: when an invalid `dimensions` argument is
            provided.
        """

        # verbose
        self.v = False

        if dimensions == 2:
            self.mesh_elements = _mesh_elements_2d
            self.family_volume = np.pi*np.power(horizon, 2)
        elif dimensions == 3:
            self.mesh_elements = _mesh_elements_3d
            self.family_volume = (4./3)*np.pi*np.power(horizon, 3)
        else:
            raise DimensionalityError(dimensions)

        self.dimensions = dimensions
        self.degrees_freedom = 3

        # bb515 Are the stiffness correction factors calculated using mesh element
        # volumes (default 'precise', 1) or average nodal volume of a transfinite
        # mesh (0)      
        self.precise_stiffness_correction = precise_stiffness_correction
        # bb515 Is the mesh transfinite mesh (support regular grid spacing with 
        # cuboidal (not tetra) elements, look up "gmsh transfinite") (default 0)
        # I'm only planning on using this for validation against literature
        self.transfinite = transfinite
        # Peridynamics parameters. These parameters will be passed to openCL
        # kernels by command line argument Bond-based peridynamics, known in
        # PDLAMMPS as Prototype Microelastic Brittle (PMB) Model requires a
        #poisson ratio of v = 0.25, but this makes little to no difference 
        # in quasi-brittle materials
        self.poisson_ratio = 0.25
        # These are the parameters that the user needs to define
        self.density = density
        self.horizon = horizon
        self.damping = damping
        self.dx = dx
        self.sigma = sigma
        self.l = l
        self.bond_stiffness_const = bond_stiffness_const
        self.critical_stretch_const = critical_stretch_const
        self.crack_length = crack_length
        self.volume_total = volume_total
        self.network_file_name = network_file_name
        self.dt = None
        self.max_reaction = None
        self.load_scale_rate = None

        self._read_mesh(mesh_file_name)

        st = time.time()

        self._set_volume(volume_total)
        # Set covariance matrix
        self._set_H(l, sigma, bond_stiffness_const, critical_stretch_const)
        # If the network has already been written to file, then read, if not, setNetwork
        try:
            self._read_network(network_file_name)
        except:
            print('No network file found: writing network file.')
            self._set_network(self.horizon, bond_type)

        # Initate crack
        self._set_connectivity(initial_crack)

        if self.v == True:
            print(
                "Building horizons took {} seconds. Horizon length: {}".format(
                    (time.time() - st), self.max_horizon_length))

        # Initiate boundary condition containers
        self.bc_types = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.intc)
        self.bc_values = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.float64)
        self.tip_types = np.zeros(self.nnodes, dtype=np.intc)
        
        if self.v == True:
            print("sum total volume", self.sum_total_volume)
            print("user input volume total", volume_total)


    def _set_network(self, horizon, bond_type):
        """
        Sets the family matrix, and converts this to a horizons matrix 
        (a fixed size data structure compatible with OpenCL).
        Calculates horizons_lengths
        Also initiate crack here if there is one
        :arg horizon: Peridynamic horizon distance
        :returns: None
        :rtype: NoneType
        """
        def l2(y1, y2):
            """
            Euclidean distance between nodes y1 and y2.
            """
            l2 = 0
            for i in range(len(y1)):
                l2 += (y1[i] - y2[i]) * (y1[i] - y2[i])
            l2 = np.sqrt(l2)
            return l2

        # Container for nodal family
        family = []
        bond_stiffness_family = []
        bond_critical_stretch_family = []

        # Container for number of nodes (including self) that each of the nodes
        # is connected to
        self.horizons_lengths = np.zeros(self.nnodes, dtype=np.intc)

        for i in range(0, self.nnodes):
            print('node', i, 'networking...')
            # Container for family nodes
            tmp = []
            # Container for bond stiffnesses
            tmp2 = []
            # Container for bond critical stretches
            tmp3 = []
            for j in range(0, self.nnodes):
                if i != j:
                    distance = l2(self.coords[i, :], self.coords[j, :])
                    if distance < horizon:
                        tmp.append(j)
                        # Determine the material properties for that bond
                        material_flag = bond_type(self.coords[i, :], self.coords[j, :])
                        if material_flag == 'steel':
                            tmp2.append(1.0) # stiffness multiplier
                            tmp3.append(1e10) # critical strain of no-fail zone, a large number
                        elif material_flag == 'interface':
                            tmp2.append(1.0) # In the probabilistic examples, "steel" is used for a no-fail-zone to prevent boundary effects
                            tmp3.append(1.0)
                        elif material_flag == 'concrete':
                            tmp2.append(1.0)
                            tmp3.append(1.0)

            family.append(np.zeros(len(tmp), dtype=np.intc))
            bond_stiffness_family.append(np.zeros(len(tmp2), dtype=np.float64))
            bond_critical_stretch_family.append(np.zeros(len(tmp3), dtype=np.float64))

            self.horizons_lengths[i] = np.intc((len(tmp)))
            for j in range(0, len(tmp)):
                family[i][j] = np.intc(tmp[j])
                bond_stiffness_family[i][j] = np.float64(tmp2[j])
                bond_critical_stretch_family[i][j] = np.float64(tmp3[j])

        assert len(family) == self.nnodes
        # As numpy array
        self.family = np.array(family)

        # Do the bond critical ste
        self.bond_critical_stretch_family = np.array(bond_critical_stretch_family)
        self.bond_stiffness_family = np.array(bond_stiffness_family)

        self.family_v = np.zeros(self.nnodes)
        for i in range(0, self.nnodes):
            tmp = 0 # tmp family volume
            family_list = family[i]
            for j in range(0, len(family_list)):
                tmp += self.V[family_list[j]]
            self.family_v[i] = tmp

        if self.precise_stiffness_correction == 1:
            # Calculate stiffening factor nore accurately using actual nodal volumes
            for i in range(0, self.nnodes):
                family_list = family[i]
                nodei_family_volume = self.family_v[i]
                for j in range(len(family_list)):
                    nodej_family_volume = self.family_v[j]
                    stiffening_factor = 2.* self.family_volume /  (nodej_family_volume + nodei_family_volume)
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 0:
            # TODO: check this code, it was 23:52pm
            average_node_volume = self.volume_total/self.nnodes
            # Calculate stiffening factor - surface corrections for 2D/3D problem, for this we need family matrix
            for i in range(0, self.nnodes):
                nnodes_i_family = len(family[i])
                nodei_family_volume = nnodes_i_family * average_node_volume # Possible to calculate more exactly, we have the volumes for free
                for j in range(len(family[i])):
                    nnodes_j_family = len(family[j])
                    nodej_family_volume = nnodes_j_family* average_node_volume # Possible to calculate more exactly, we have the volumes for free
                    
                    stiffening_factor = 2.* self.family_volume /  (nodej_family_volume + nodei_family_volume)
                    
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 2:
            # Don't apply stiffness correction factor
            pass

        # Maximum number of nodes that any one of the nodes is connected to, must be a power of 2 (for OpenCL reduction)
        self.max_horizon_length = np.intc(
                    1<<(len(max(family, key=lambda x: len(x)))-1).bit_length()
                )

        self.horizons = -1 * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.family):
            self.horizons[i][0:len(j)] = j

        self.bond_stiffness = -1. * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.bond_stiffness_family):
            self.bond_stiffness[i][0:len(j)] = j

        self.bond_critical_stretch = -1. * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.bond_critical_stretch_family):
            self.bond_critical_stretch[i][0:len(j)] = j

        # Make sure it is in a datatype that C can handle
        self.horizons = self.horizons.astype(np.intc)

        vtk.writeNetwork(self.network_file_name, "Network",
                      self.max_horizon_length, self.horizons_lengths,
                      self.family, self.bond_stiffness_family, self.bond_critical_stretch_family)

    def _set_H(self, l, sigma, bond_stiffness_const, critical_stretch_const, epsilon=1e-5):
        """
        Constructs the failure strains matrix and H matrix, which is a sparse
        matrix containing distances.
        :returns: None
        :rtype: NoneType
        """
        # Set model parameters
        self.bond_stiffness_const = bond_stiffness_const
        self.critical_stretch_const = critical_stretch_const
        # TODO: How much of this could be done in OpenCL?
        # These are all element-wise operations so cost is with O(n^2), so not too bad
        # Apart from choleshky decomp
        coords = self.coords

        # Extract the coordinates
        V_x = coords[:, 0]
        V_y = coords[:, 1]
        V_z = coords[:, 2]

        # Tiled matrices
        lam_x = np.tile(V_x, (self.nnodes, 1))
        lam_y = np.tile(V_y, (self.nnodes, 1))
        lam_z = np.tile(V_z, (self.nnodes, 1))

        # Dense matrices
        H_x0 = -lam_x + lam_x.transpose()
        H_y0 = -lam_y + lam_y.transpose()
        H_z0 = -lam_z + lam_z.transpose()

        norms_matrix = (
                np.power(H_x0, 2) + np.power(H_y0, 2) + np.power(H_z0, 2)
                )
        # inv length scale parameter
        inv_length_scale = np.divide(-1., 2.*pow(l,2))
        # radial basis functions
        rbf = np.multiply(inv_length_scale, norms_matrix)

        # Exponential of radial basis functions, Covariance matrix
        self.K = np.exp(rbf)

        # Create C matrix for sampling perturbations:
        # add epsilon, a numerical trick so that is semi postive definite
        I = np.identity(self.nnodes)
        K_tild = np.add(self.K, np.multiply(epsilon, I))
        K_tild = np.multiply(pow(sigma, 2), K_tild)
        # Cholesky decomposition
        C = np.linalg.cholesky(2*K_tild)
        # Multiply by a standard deviation (vertical scale), sigma
        self.C = np.multiply (pow(sigma, 2), C)

    def simulate(self, model, sample, realisation, steps, integrator, write=None, toolbar=0, 
                 displacement_rate=None, build_displacement=None, final_displacement=None):
        """
        Simulate the peridynamics model.
        :arg int steps: The number of simulation steps to conduct.
        :arg  integrator: The integrator to use, see
            :mod:`peridynamics.integrators` for options.
        :type integrator: :class:`peridynamics.integrators.Integrator`
        :arg boundary_function: A function to apply the boundary conditions for
            the simlation. It has the form
            boundary_function(:class:`peridynamics.model.Model`,
            :class:`numpy.ndarray`, `int`). The arguments are the model being
            simulated, the current displacements, and the current step number
            (beginning from 1). `boundary_function` returns a (nnodes, 3)
            :class:`numpy.ndarray` of the updated displacements
            after applying the boundary conditions. Default `None`.
        :type boundary_function: function
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling
            :meth:`peridynamics.model.Model.write_mesh`. If `None` then no
            output is written. Default `None`.
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)
        # Calculate number of time steps that displacement load is in the 'build-up' phase
        if not ((displacement_rate is None) or (build_displacement is None) or (final_displacement is None)):
            build_time, a, b, c= _calc_build_time(build_displacement, displacement_rate, steps)
            
        # Container for plotting data
        damage_data = []

        # Ease off displacement loading switch
        ease_off = 0

        #Progress bar
        toolbar_width = 40

        if toolbar:    
            sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        st = time.time()
        for step in range(1, steps+1):
            # Conduct one integration step
            integrator.runtime(model, step-1)
            if write:
                if step % write == 0:
                    damage_data = integrator.write(model, step, sample, realisation)
                    if toolbar == 0:
                        print('Print number {}/{} complete in {} s '.format(int(step/write), int(steps/write), time.time() - st))
                        st = time.time()
            
            # Increase load in linear increments
            if not (model.load_scale_rate is None):
                load_scale = min(1.0, model.load_scale_rate * step)
                if load_scale != 1.0:
                    integrator.incrementLoad(model, load_scale)
            # Increase dispalcement in 5th order polynomial increments
            if not ((displacement_rate is None) or (build_displacement is None) or (final_displacement is None)):
                # 5th order polynomial/ linear curve used to calculate displacement_scale
                displacement_scale, ease_off = _calc_load_displacement_rate(a, b, c,
                                                                 final_displacement,
                                                                 build_time,
                                                                 displacement_rate,
                                                                 step, 
                                                                 build_displacement,
                                                                 ease_off)
                if displacement_scale != 0.0:
                    integrator.incrementDisplacement(model, displacement_scale)
            # No user specified build up parameters case
            elif not (displacement_rate is None):
                integrator.incrementDisplacement(model, 1.0)

            # Loading bar update
            if step%(steps/toolbar_width)<1 & toolbar:
                sys.stdout.write("\u2588")
                sys.stdout.flush()

        if toolbar:
            sys.stdout.write("]\n")

        return damage_data

def initial_crack_helper(crack_function):
    """
    Help the construction of an initial crack function.
    `crack_function` has the form `crack_function(icoord, jcoord)` where
    `icoord` and `jcoord` are :class:`numpy.ndarray` s representing two node
    coordinates.  crack_function returns a truthy value if there is a crack
    between the two nodes and a falsy value otherwise.
    This decorator returns a function which takes all node coordinates and
    returns a list of tuples of the indices pair of nodes which define the
    crack. This function can therefore be used as the `initial_crack` argument
    of the :class:`Model`
    :arg function crack_function: The function which determine whether there is
        a crack between a pair of node coordinates.
    :returns: A function which determines all pairs of nodes with a crack
        between them.
    :rtype: function
    """
    def initial_crack(coords, neighbourhood):
        crack = []
        # Get all pairs of bonded particles (within horizon) where i < j and
        # i /= j (using the upper triangular portion)
        i, j = sparse.triu(neighbourhood, 1).nonzero()
        # Check each pair using the crack function
        for i, j in zip(i, j):
            if crack_function(coords[i], coords[j]):
                crack.append((i, j))
        return crack
    return initial_crack

def _calc_midpoint_gradient(T, displacement_scale_rate):
    A = np.array([
        [(1*T**5)/1,(1*T**4)/1,(1*T**3)/1],
        [(20*T**3)/1,(12*T**2)/1,(6*T**1)/1],
        [(5*T**4)/1,(4*T**3)/1,(3*T**2)/1,]
        ]
        )
    b = np.array(
        [
            [displacement_scale_rate],
            [0.0],
            [0.0]
                ])
    x = np.linalg.solve(A,b)
    a = x[0][0]

    b = x[1][0]

    c = x[2][0]
    
    midpoint_gradient = (5./16)*a*T**4 + (4./8)*b*T**3 + (3./4)*c*T**2
    
    return(midpoint_gradient, a, b, c)

def _calc_build_time(build_displacement, displacement_scale_rate, steps):
    T = 0
    midpoint_gradient = np.inf
    while midpoint_gradient > displacement_scale_rate:
        try:
            midpoint_gradient, a, b, c = _calc_midpoint_gradient(T, build_displacement)
        except:
            pass
        T += 1
        if T > steps:
            # TODO: suggest some valid values from the parameters given
            raise ValueError('Displacement build-up time was larger than total simulation time steps! \ntry decreasing build_displacement, or increase max_displacement_rate. steps = {}'.format(steps))
            break
    return(T, a, b, c)

def _calc_load_displacement_rate(a, b, c, final_displacement, build_time, displacement_scale_rate, step, build_displacement, ease_off):
    if step < build_time/2:
        m = 5*a*step**4 + 4*b*step**3 + 3*c*step**2
        #print('m = ', m)
        load_displacement_rate = m/displacement_scale_rate
    elif ease_off != 0:
        t = step - ease_off + build_time/2
        if t > build_time:
            load_displacement_rate = 0.0
        else:
            m = 5*a*t**4 + 4*b*t**3 + 3*c*t**2
            load_displacement_rate = m/displacement_scale_rate
    else: # linear regime
        # calculate displacement
        linear_regime_time = step - build_time/2
        linear_regime_displacement = linear_regime_time * displacement_scale_rate
        displacement = linear_regime_displacement + build_displacement/2
        if displacement + build_displacement/2 < final_displacement:
            load_displacement_rate = 1.0
        else:
            ease_off = step
            load_displacement_rate = 1.0
    return(load_displacement_rate, ease_off)

class DimensionalityError(Exception):
    """An invalid dimensionality argument used to construct a model."""

    def __init__(self, dimensions):
        """
        Construct the exception.
        :arg int dimensions: The number of dimensions passed as an argument to
            :meth:`Model`.
        :rtype: :class:`DimensionalityError`
        """
        message = (
            f"The number of dimensions must be 2 or 3,"
            " {dimensions} was given."
            )

        super().__init__(message)


class InvalidIntegrator(Exception):
    """An invalid integrator has been passed to `simulate`."""

    def __init__(self, integrator):
        """
        Construct the exception.
        :arg integrator: The object passed to :meth:`Model.simulate` as the
            integrator argument.
        :rtype: :class:`InvalidIntegrator`
        """
        message = (
            f"{integrator} is not an instance of"
            "peridynamics.integrators.Integrator"
            )

        super().__init__(message)