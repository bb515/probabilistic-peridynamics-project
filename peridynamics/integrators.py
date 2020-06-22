"""Integrators."""
from abc import ABC, abstractmethod
import pyopencl as cl
import numpy as np
import sys
import pathlib
from peridynamics.post_processing import vtk
sys.path.insert(1, pathlib.Path(__file__).parent.absolute() / 'peridynamics/kernels')

class Integrator(ABC):
    """
    Base class for integrators.

    All integrators must define a call method which performs one
    integration step and returns the updated displacements.
    """

    @abstractmethod
    def __call__(self):
        """
        Conduct one iteraction of the integrator.

        This method should be implemennted in every concrete integrator.
        """
class Euler(Integrator):
    r"""
    Euler integrator.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """

    def __init__(self, dt, dampening=1.0):
        """
        Create an :class:`Euler` integrator object.

        :arg float dt: The integration time step.
        :arg float dampening: The dampening factor. The default is 1.0

        :returns: A :class:`Euler` object
        """
        self.dt = dt
        self.dampening = dampening

    def __call__(self, u, f):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """
        return u + self.dt * f * self.dampening
            
class RK4(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_RK4.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un3 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un3 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un, self.d_un1)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                      None, self.d_k2dn, self.d_un, self.d_un2)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k3dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update2(self.queue, 
                                                   (model.nnodes * model.degrees_freedom,),
                                                   None, self.d_k3dn, self.d_un, self.d_un3)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k4dn, self.d_un3, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Finally update the displacements using weighted average of 4 incriments
        self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_bc_types, self.d_bc_values, self.d_un)
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)

class RK4Optimised(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_RK4_optimised.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        # Not needed, as CheckBonds is done in CalcBondForce
        #self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_reduce_force = program.ReduceForce
        self.cl_kernel_reduce_damage = program.ReduceDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un3 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Bond forces
        self.h_forces =  np.empty((model.nnodes, model.degrees_freedom, model.max_horizon_length), dtype=np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un3 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_forces = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_forces.nbytes)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)

        # Initialize kernel parameters
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None])
        #self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_reduce_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes(
            [None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Calc bond forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k1dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un, self.d_un1)
        # Calc bond forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k2dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                      None, self.d_k2dn, self.d_un, self.d_un2)
        # Calc bond forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k3dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update2(self.queue, 
                                                   (model.nnodes * model.degrees_freedom,),
                                                   None, self.d_k3dn, self.d_un, self.d_un3)
        # Calc bond forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un3, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k4dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Finally update the displacements using weighted average of 4 incriments
        self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_bc_types, self.d_bc_values, self.d_un)
        # Check for broken bonds
        # Not needed, as CheckBonds is done in CalcBondForce
        #self.cl_kernel_check_bonds(self.queue,
                              #(model.nnodes, model.max_horizon_length),
                              #None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)

class DormandPrince(Integrator):
    r"""
    4th order Runge-Kutta with adaptive time step

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model, error_size_max=1e-7, error_size_min=1e-10):
        """ Initialise the integration scheme
        """
        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_dormand_prince.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_partial_displacement_update3 = program.PartialUpdateDisplacement3
        self.cl_kernel_partial_displacement_update4 = program.PartialUpdateDisplacement4
        self.cl_kernel_partial_displacement_update5 = program.PartialUpdateDisplacement5
        self.cl_kernel_partial_displacement_update6 = program.PartialUpdateDisplacement6
        self.cl_kernel_check_error = program.CheckError

        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set local variables
        self.error_size_max = error_size_max
        self.error_size_min = error_size_min

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un4 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un5 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un5_1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un_temp = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k5dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k6dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k7dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un4 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un4.nbytes)
        self.d_un5 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un5.nbytes)
        self.d_un5_1= cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un5_1.nbytes)
        self.d_un_temp = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un_temp.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k4dn.nbytes)
        self.d_k5dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k5dn.nbytes)
        self.d_k6dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k6dn.nbytes)
        self.d_k7dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k7dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update3.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update4.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update5.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update6.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])

    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un5, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements 2
        self.cl_kernel_partial_displacement_update2(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k3dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements 3
        self.cl_kernel_partial_displacement_update3(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k4dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k4dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements 4
        self.cl_kernel_partial_displacement_update4(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k5dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k5dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements 5
        self.cl_kernel_partial_displacement_update5(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k6dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k6dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Partial update of displacements 6
        self.cl_kernel_partial_displacement_update6(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_k6dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k7dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k7dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        if self.adapt_time_step(model) == 1:
            pass
        else:
            # Full update of displacements
            self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_bc_types, self.d_bc_values, self.d_un_temp, self.d_un5, self.h_dt)
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un5, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un5, self.d_un5)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un5[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un5)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force
    def adapt_time_step(self, model):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_k1dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_k6dn, self.d_k7dn, self.d_un_temp, self.d_un5, self.d_errorn, self.h_dt)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > self.error_size_max:
            self.h_dt /= 1.1
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < self.error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)

class DormandPrinceOptimised(Integrator):
    r"""
    4th order Runge-Kutta with adaptive time step

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model, error_size_max=1e-7, error_size_min=1e-10):
        """ Initialise the integration scheme
        """
        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_dormand_prince_optimised.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_partial_displacement_update3 = program.PartialUpdateDisplacement3
        self.cl_kernel_partial_displacement_update4 = program.PartialUpdateDisplacement4
        self.cl_kernel_partial_displacement_update5 = program.PartialUpdateDisplacement5
        self.cl_kernel_partial_displacement_update6 = program.PartialUpdateDisplacement6
        self.cl_kernel_check_error = program.CheckError
        self.cl_kernel_reduce_force = program.ReduceForce
        self.cl_kernel_reduce_damage = program.ReduceDamage
        self.cl_kernel_check_bonds = program.CheckBonds

        # Set local variables
        self.error_size_max = error_size_max
        self.error_size_min = error_size_min

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un4 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un5 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un5_1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un_temp = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k5dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k6dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k7dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Bond forces
        self.h_forces =  np.empty((model.nnodes, model.degrees_freedom, model.max_horizon_length), dtype=np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un4 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un4.nbytes)
        self.d_un5 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un5.nbytes)
        self.d_un5_1= cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un5_1.nbytes)
        self.d_un_temp = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un_temp.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k4dn.nbytes)
        self.d_k5dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k5dn.nbytes)
        self.d_k6dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k6dn.nbytes)
        self.d_k7dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k7dn.nbytes)
        self.d_forces = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_forces.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update3.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update4.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update5.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update6.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_reduce_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes(
            [None, None, None, None])

    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Calc bond forces k1dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un5, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k1dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Calc bond forces k2dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k2dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements 2
        self.cl_kernel_partial_displacement_update2(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Calc bond forces k3dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k3dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements 3
        self.cl_kernel_partial_displacement_update3(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Calc bond forces k4dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k4dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements 4
        self.cl_kernel_partial_displacement_update4(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Calc bond forces k5dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k5dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements 5
        self.cl_kernel_partial_displacement_update5(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Calc bond forces k6dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k6dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Partial update of displacements 6
        self.cl_kernel_partial_displacement_update6(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_k6dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Calc bond forces k7dn
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k7dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        if self.adapt_time_step(model) == 1:
            pass
        else:
            # Full update of displacements
            self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_bc_types, self.d_bc_values, self.d_un_temp, self.d_un5, self.h_dt)
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un5, self.d_coords, self.d_bond_critical_stretch)
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un5, self.d_un5)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un5[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un5)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force
    def adapt_time_step(self, model):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_k1dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_k6dn, self.d_k7dn, self.d_un_temp, self.d_un5, self.d_errorn, self.h_dt)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > self.error_size_max:
            self.h_dt /= 1.1
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < self.error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt
    

class EulerStochastic(Integrator):
    r"""
    Stochastic Euler integrator for quasi-static loading, using optimised OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_stochastic.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_mmul = program.mmul
        self.cl_kernel_update_displacement = program.UpdateDisplacement
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_reduce_force = program.ReduceForce
        self.cl_kernel_reduce_damage = program.ReduceDamage
        self.cl_kernel_matrix_vector_mul1 = program.gemv1
        self.cl_kernel_matrix_vector_mul2 = program.gemv2

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Bond forces
        self.h_forces =  np.empty((model.nnodes, model.degrees_freedom, model.max_horizon_length), dtype=np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Covariance matrix
        self.h_K = model.K

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        # 
        self.h_m = np.intc(
        1<<(model.nnodes-1).bit_length()
        )
        self.h_n = np.intc(model.nnodes)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)
        self.d_K = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_K)
        

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        self.d_udn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        self.d_forces = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_forces.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_reduce_force.set_scalar_arg_dtypes([None, None, None, None, None, None])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_mmul.set_scalar_arg_dtypes([None, None, None])
        self.cl_kernel_matrix_vector_mul1.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_matrix_vector_mul2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """
    def noise(self, C, K, num_nodes, num_steps, degrees_freedom = 3):
        """Takes sample from multivariate normal distribution 
        with covariance matrix whith Cholesky factor, L
        :arg L: Cholesky factor, C
        :arg C: Covariance matrix, K
        :arg samples: The number of degrees of freedom (read: dimensions) the
        noise is generated in, degault 3 i.e. x,y and z directions.
        :returns: num_nodes * 3 * num_steps array of noise
        :rtype: np.array dtype=float64
        """

        def multivar_normal(self, L, num_nodes):
            """ Fn for taking a single multivar normal sample covariance matrix with Cholesky factor, L
            """
            # Pad L
            shape = np.shape(L)
            padded_L = np.zeros((self.h_m, self.h_n))
            padded_L[:shape[0],:shape[1]] = L

            # OpenCL kernel reads L in column major not row major order
            h_L = np.ascontiguousarray(np.transpose(padded_L), dtype=np.float64)

            h_x = np.ascontiguousarray(np.random.normal(0, 1, size = num_nodes), dtype=np.float64)
            h_y = np.empty((self.h_n), dtype=np.float64)
            # Read only
            d_x = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=h_x)
            d_L = cl.Buffer(self.context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=h_L)
            # Write only
            d_y = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_y.nbytes)
            self.cl_kernel_matrix_vector_mul1(self.queue, (self.h_m,), (128,),
                            d_L, d_x, d_y, self.h_m, self.h_n)
            # Device to host
            cl.enqueue_copy(self.queue, h_y, d_y)
            # CPU version
            #y = np.dot(L, h_x) #vector
            #zeros = np.subtract(h_y, y)
            #error = abs(np.max(zeros))
            #print(error)
            #assert (error < 1e-13), 'error was too large, something is wrong, error was {}'.format(error)
            print('noise step complete')
            return h_y

        noise = []
        for i in range(degrees_freedom * num_steps):
            noise.append(multivar_normal(self, C, num_nodes))
        return np.ascontiguousarray(noise, dtype=np.float64)

    def reset(self, model, steps):
        # Displacements
        self.h_un = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        #self.h_udn1 = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.zeros(model.nnodes).astype(np.float64)

        # Sample random noise vector using openCL
        #self.h_pn = noise(model.C, model.K, model.nnodes, steps)
        self.h_pn = self.noise(model.C, model.K, model.nnodes, steps)

        # Covariance matrix
        #self.h_K = model.K

        # Build OpenCL data structures

        # Read only
        self.d_pn = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_pn)
        #self.d_K = cl.Buffer(
                #self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                #hostbuf=self.h_K)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_un)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn)
        #self.d_udn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_damage)
        # Initialize kernel parameters
    def runtime(self, model, step):
        # Time marching Part 1
        self.cl_kernel_update_displacement(self.queue, (model.nnodes* model.degrees_freedom,),
                                  None, self.d_udn, self.d_un, self.d_pn, self.d_bc_types,
                                  self.d_bc_values, np.intc(step))
        # Time marching Part 2: Calc forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_udn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Covariance matrix multiplication of forces
        #self.cl_kernel_mmul(self.queue, (model.nnodes, model.degrees_freedom), None, self.d_K, self.d_udn, self.d_udn1)
    def write(self, model, t, sample, realisation):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        vtk.write("output/U_"+"sample" + str(sample) +"realisation" +str(realisation) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        vtk.writeDamage("output/damage_" + "sample" + str(sample)+ "realisation" +str(realisation) + ".vtk", "Title", self.h_damage)
        return self.h_damage

class HeunEuler(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model, error_size_max=1e-2, error_size_min=1e-10):
        """ Initialise the integration scheme
        """
        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_heun_euler.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage
        self.cl_kernel_check_error = program.CheckError

        # Input arguments
        self.error_size_max = error_size_max
        self.error_size_min = error_size_min

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        # First order accurate displacement
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement update delta
        self.h_un2_1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un1.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_un2_1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None, None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        # Find first order accurate displacements
        # Scalars like self.h_dt can live on the host memory
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un2, self.d_un1, self.h_dt)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values, self.h_force_load_scale)
        if self.adapt_time_step(model) == 1:
            # Do not update second order displacements (i.e. repeat time step)
            pass
        else:
            # Update second order displacements
            self.cl_kernel_displacement_update(self.queue,
                                               (model.nnodes * model.degrees_freedom,), 
                                               None, self.d_bc_types, self.d_bc_values, self.d_un2_1, self.d_un2, self.h_dt)
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un2, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un2, self.d_un2)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un2[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un2)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force
    def adapt_time_step(self, model):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_k1dn, self.d_k2dn, self.d_un1, self.d_un2, self.d_un2_1, self.d_errorn, self.h_dt)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > self.error_size_max:
            self.h_dt /= 1.1
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < self.error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)

class HeunEulerOptimised(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model, error_size_max=1e-2, error_size_min=1e-10):
        """ Initialise the integration scheme
        """

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_heun_euler_optimised.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_update_displacement = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_check_error = program.CheckError
        self.cl_kernel_reduce_force = program.ReduceForce
        self.cl_kernel_reduce_damage = program.ReduceDamage

        # Input arguments
        self.error_size_max = error_size_max
        self.error_size_min = error_size_min

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        # First order accurate displacement
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement update delta
        self.h_un2_1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Bond forces
        self.h_forces =  np.empty((model.nnodes, model.degrees_freedom, model.max_horizon_length), dtype=np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un1.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_un2_1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_forces = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_forces.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_reduce_force.set_scalar_arg_dtypes([None, None, None, None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None, None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Calc bond forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,), 
                                    (model.max_horizon_length,), self.d_forces, self.d_k1dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        # Find first order accurate displacements
        # Scalars like self.h_dt can live on the host memory
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un2, self.d_un1, self.h_dt)
        # Calc bond forces
        self.cl_kernel_calc_bond_force(self.queue, (model.nnodes, model.max_horizon_length), None, self.d_forces,
                                  self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_reduce_force(self.queue, (model.max_horizon_length * model.degrees_freedom * model.nnodes,),
                                  (model.max_horizon_length,), self.d_forces, self.d_k2dn, self.d_force_bc_types, self.d_force_bc_values, self.local_mem, self.h_force_load_scale)
        if self.adapt_time_step(model) == 1:
            # Do not update second order displacements (i.e. repeat time step)
            pass
        else:
            # Update second order displacements
            self.cl_kernel_update_displacement(self.queue,
                                               (model.nnodes * model.degrees_freedom,), 
                                               None, self.d_bc_types, self.d_bc_values, self.d_un2_1, self.d_un2, self.h_dt)
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un2, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un2, self.d_un2)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un2[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un2)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force
    def adapt_time_step(self, model):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_k1dn, self.d_k2dn, self.d_un1, self.d_un2, self.d_un2_1, self.d_errorn, self.h_dt)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > self.error_size_max:
            self.h_dt /= 1.1
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < self.error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)

class EulerCromerOptimisedLumped2(Integrator):
    r"""
    Dynamic Euler integrator using OpenCL kernels.

    The Euler-Cromer method is a first-order, dynamic (acceleration term is not neglected), numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t v(t)
        v(t + \delta t) = v(t) + \delta t a(t)

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`v(t)` is
    the velocity at time :math:`t`, :math:`a(t)` is the acceleration at time :math:`t`,
    :math:`\delta t` is the time step.
    """
    def __init__(self, model):
        """ Initialise the integration scheme for Euler Cromer
        """
        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_cromer_optimised_2.2.cl").read()

        # JIT compiler's command line arguments
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP
            + "-DPD_DX=" + str(model.dx) + SEP
            + "-DPD_R=" + str(model.horizon) + SEP)

        # Build the programs
        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_update_acceleration = program.UpdateAcceleration
        self.cl_kernel_update_velocity = program.UpdateVelocity
        self.cl_kernel_update_displacement = program.UpdateDisplacement
        self.cl_kernel_reduce_damage = program.ReduceDamage

        # Set initial values in host memory
        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)
        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values
        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V
        # Bond stiffnesses
        self.h_bond_stiffness = np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacement
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Velocity
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Acceleration
        self.h_uddn = np.empty((model.nnodes, model.degrees_freedom), dtype = np.float64)
        # Node Forces
        self.h_node_forces = np.empty((model.nnodes, model.degrees_freedom), dtype = np.float64)

        # Bond forces
        self.local_mem_x = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_y = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_z = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # For measuring tip displacemens (host memory only)
        self.h_tip_types = model.tip_types

        # For applying force incrementally
        self.h_force_load_scale = np.float64(0.0)
        self.h_displacement_load_scale = np.float64(0.0)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("udn", self.h_udn.dtype)
            print("damage", self.h_damage.dtype)
            print("stiffness", self.h_bond_stiffness.dtype)
            print("stretch", self.h_bond_critical_stretch.dtype)

        # Build OpenCL data structures
        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn.nbytes)
        self.d_uddn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_uddn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_node_forces = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_node_forces.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_update_acceleration.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
        self.cl_kernel_update_velocity.set_scalar_arg_dtypes(
            [None, None])
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes([None, None, None, None])
        return None

    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        """ Run time integration for Euler Cromer scheme
        """
        # Update displacements
        self.cl_kernel_update_displacement(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_un, self.d_bc_types,
                                  self.d_bc_values, self.h_displacement_load_scale)
        # Reduction of bond forces onto nodal forces
        self.cl_kernel_update_acceleration(
                self.queue, (model.max_horizon_length * model.nnodes,), (model.max_horizon_length,),
                self.d_un,
                self.d_udn,
                self.d_uddn,
                self.d_node_forces,
                self.d_vols,
                self.d_horizons,
                self.d_coords,
                self.d_bond_stiffness,
                self.d_bond_critical_stretch,
                self.d_force_bc_types,
                self.d_force_bc_values,
                self.local_mem_x,
                self.local_mem_y,
                self.local_mem_z,
                self.h_force_load_scale)
        # Update velocity
        self.cl_kernel_update_velocity(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_uddn)

    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        cl.enqueue_copy(self.queue, self.h_uddn, self.d_uddn)
        cl.enqueue_copy(self.queue, self.h_node_forces, self.d_node_forces)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_acceleration = 0
        tip_force = 0
        sum_damage = []
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][0]
                tip_acceleration += self.h_uddn[i][0]
                tip_force += self.h_node_forces[i][0] * model.V[i]
                sum_damage.append(np.sum(self.h_damage))
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output2/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_force

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)

class EulerOpenCLMCMC(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_optimised_2.3.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_integration = program.TimeIntegration
        self.cl_kernel_update_displacement = program.UpdateDisplacement
        self.cl_kernel_reduce_damage = program.ReduceDamage

        # Set initial values in host memory
        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Constants
        self.h_bond_stiffness_const = np.float64(model.bond_stiffness_const)
        self.h_bond_critical_stretch_const = np.float64(model.critical_stretch_const)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Bond forces
        self.local_mem_x = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_y = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_z = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)
        # For applying displacement in incriments
        self.h_displacement_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn.nbytes)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_integration.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None
             ])
        # Initialize kernel parameters
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None
             ])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes(
            [None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def reset(self, model):
        # Displacements
        self.h_un = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        self.h_udn = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        
        # Update parameters
        self.h_bond_stiffness_const = np.float64(model.bond_stiffness_const)
        self.h_bond_critical_stretch_const= np.float64(model.critical_stretch_const)

        # Damage vector
        self.h_damage = np.zeros(model.nnodes).astype(np.float64)

        # Horizons
        self.h_horizons = model.horizons

            # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_un)

        # NOTE we must use COPY_HOST_PTR here
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_damage)

    def runtime(self, model):
        # Update displacements
        self.cl_kernel_update_displacement(
                self.queue, (model.degrees_freedom * model.nnodes,), None,
                self.d_udn,
                self.d_un,
                self.d_bc_types,
                self.d_bc_values,
                self.h_displacement_load_scale
                )
        # Time integration step
        self.cl_kernel_time_integration(
                self.queue, (model.nnodes * model.max_horizon_length,), (model.max_horizon_length,), 
                self.d_un,
                self.d_udn,
                self.d_vols,
                self.d_horizons,
                self.d_coords,
                self.d_bond_stiffness,
                self.d_bond_critical_stretch,
                self.d_force_bc_types,
                self.d_force_bc_values,
                self.local_mem_x,
                self.local_mem_y,
                self.local_mem_z,
                self.h_force_load_scale,
                self.h_displacement_load_scale,
                self.h_bond_stiffness_const,
                self.h_bond_critical_stretch_const
                )

    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        cl.enqueue_copy(self.queue, self.h_udn, self.d_udn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
                tip_shear_force += self.h_udn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) + "t" + str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)
        
class EulerOpenCLOptimisedLumped2(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_optimised_2.2.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_integration = program.TimeIntegration
        self.cl_kernel_update_displacement = program.UpdateDisplacement
        self.cl_kernel_reduce_damage = program.ReduceDamage

        # Set initial values in host memory
        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Bond forces
        self.local_mem_x = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_y = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_z = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)
        # For applying displacement in incriments
        self.h_displacement_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn.nbytes)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_integration.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None
             ])
        # Initialize kernel parameters
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None
             ])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes(
            [None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Update displacements
        self.cl_kernel_update_displacement(
                self.queue, (model.degrees_freedom * model.nnodes,), None,
                self.d_udn,
                self.d_un,
                self.d_bc_types,
                self.d_bc_values,
                self.h_displacement_load_scale
                )
        # Time integration step
        self.cl_kernel_time_integration(
                self.queue, (model.nnodes * model.max_horizon_length,), (model.max_horizon_length,), 
                self.d_un,
                self.d_udn,
                self.d_vols,
                self.d_horizons,
                self.d_coords,
                self.d_bond_stiffness,
                self.d_bond_critical_stretch,
                self.d_force_bc_types,
                self.d_force_bc_values,
                self.local_mem_x,
                self.local_mem_y,
                self.local_mem_z,
                self.h_force_load_scale,
                self.h_displacement_load_scale
                )

    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        cl.enqueue_copy(self.queue, self.h_udn, self.d_udn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
                tip_shear_force += self.h_udn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)

class EulerStochasticOptimised(Integrator):
    r"""
    Stochastic Euler integrator for quasi-static loading, using optimised OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_stochastic_optimised.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_update_displacement = program.UpdateDisplacement
        self.cl_kernel_update_acceleration = program.UpdateAcceleration
        self.cl_kernel_reduce_damage = program.ReduceDamage
        self.cl_kernel_matrix_vector_mul1 = program.gemv1
        self.cl_kernel_matrix_vector_mul2 = program.gemv2
        self.cl_kernel_matrix_vector_mul3 = program.gemv3
        self.cl_kernel_reduce_rows = program.reduce_rows
        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses and stretch factors
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)
        
        self.h_bond_stiffness_const = np.float64(model.bond_stiffness_const)
        self.h_bond_critical_stretch_const = np.float64(model.critical_stretch_const)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn_x = np.empty((model.nnodes), dtype=np.float64)
        self.h_udn_y = np.empty((model.nnodes), dtype=np.float64)
        self.h_udn_z = np.empty((model.nnodes), dtype=np.float64)

        # Brownian motion
        self.h_bdn_x = np.empty((model.nnodes), dtype=np.float64)
        self.h_bdn_y = np.empty((model.nnodes), dtype=np.float64)
        self.h_bdn_z = np.empty((model.nnodes), dtype=np.float64)

        # Updated forces
        self.h_udn1_x = np.empty((model.nnodes), dtype=np.float64)
        self.h_udn1_y = np.empty((model.nnodes), dtype=np.float64)
        self.h_udn1_z = np.empty((model.nnodes), dtype=np.float64)

        # Updated brownian motion (sampled with a length scale)
        self.h_bdn1_x = np.empty((model.nnodes), dtype=np.float64)
        self.h_bdn1_y = np.empty((model.nnodes), dtype=np.float64)
        self.h_bdn1_z = np.empty((model.nnodes), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)
        # For applying force in incriments
        self.h_force_load_scale = np.float64(0.0)
        # For applying displacement in incriments
        self.h_displacement_load_scale = np.float64(0.0)

        # dimensions for matrix-vector multiplication
        # local (work group) size
        self.h_mdash = np.intc(16) #64
        self.h_p = np.intc(4) #16
        self.h_m = np.intc(
        1<<(model.nnodes-1).bit_length()
        )
        #self.h_n = np.intc(model.nnodes) # mvul1
        self.h_n = np.intc(np.ceil(model.nnodes/self.h_p)*self.h_p) #muvl2

        # Bond forces
        #local_mem_mvmul2 = np.empty((self.h_p * self.h_mdash), dtype=np.float64)
        self.local_mem_mvmul2_1 = cl.LocalMemory(np.dtype(np.float64).itemsize * self.h_p * self.h_mdash)
        self.local_mem_mvmul2_2 = cl.LocalMemory(np.dtype(np.float64).itemsize * self.h_p * self.h_mdash)
        self.local_mem_mvmul2_3 = cl.LocalMemory(np.dtype(np.float64).itemsize * self.h_p * self.h_mdash)
        self.local_mem_mvmul2_4 = cl.LocalMemory(np.dtype(np.float64).itemsize * self.h_p * self.h_mdash)
        self.local_mem_mvmul2_5 = cl.LocalMemory(np.dtype(np.float64).itemsize * self.h_p * self.h_mdash)
        self.local_mem_mvmul2_6 = cl.LocalMemory(np.dtype(np.float64).itemsize * self.h_p * self.h_mdash)
        self.local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_x = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_y = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)
        self.local_mem_z = cl.LocalMemory(np.dtype(np.float64).itemsize * model.max_horizon_length)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)
       

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)

        self.d_udn_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn_x.nbytes)
        self.d_udn_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn_y.nbytes)
        self.d_udn_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn_z.nbytes)

        self.d_udn1_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1_x.nbytes)
        self.d_udn1_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1_y.nbytes)
        self.d_udn1_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1_z.nbytes)

        self.d_bdn_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_bdn_x.nbytes)
        self.d_bdn_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_bdn_y.nbytes)
        self.d_bdn_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_bdn_z.nbytes)

        self.d_bdn1_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_bdn1_x.nbytes)
        self.d_bdn1_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_bdn1_y.nbytes)
        self.d_bdn1_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_bdn1_z.nbytes)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
        self.cl_kernel_update_acceleration.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_matrix_vector_mul1.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_matrix_vector_mul2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_matrix_vector_mul3.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_reduce_rows.set_scalar_arg_dtypes(
            [None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def noise(self, num_nodes, num_steps, degrees_freedom = 3):
        """Takes sample from multivariate normal distribution 
        with covariance matrix whith Cholesky factor, L
        :arg C: Cholesky factor, C
        :arg K: Covariance matrix, K
        :arg samples: The number of degrees of freedom (read: dimensions) the
        noise is generated in, degault 3 i.e. x,y and z directions.
        :returns: num_nodes * 3 * num_steps array of BROWNIAN noise
        :rtype: np.array dtype=float64
        """
        print('numpy random sample start')
        noise = np.random.normal(0, 1, size = (num_nodes, num_steps * degrees_freedom))
        print('numpy random sample end')
        return np.ascontiguousarray(noise, dtype=np.float64)

    def reset(self, model, steps):
        # Displacements
        self.h_un = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Reset initial brownian forcing and forces to 0
        self.h_udn_x = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn_y = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn_z = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1_x = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1_y = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1_z = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_bdn1_x = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_bdn1_y = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_bdn1_z = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.zeros(model.nnodes).astype(np.float64)

        # Horizons
        self.h_horizons = model.horizons

        # Gaussian noise vectors
        self.h_noises = self.noise(model.nnodes, steps)

        # Covariance matrix
        # Pad K
        shape = np.shape(model.K)
        padded_K = np.zeros((self.h_m, self.h_n))
        padded_K[:shape[0],:shape[1]] = model.K
        # OpenCL kernel reads L in column major not row major order
        self.h_K = np.ascontiguousarray(np.transpose(padded_K), dtype=np.float64)

        # Cholesky decomposition
        padded_C = np.zeros((self.h_m, self.h_n))
        padded_C[:shape[0],:shape[1]] = model.C
        # OpenCL kernel reads L in column major not row major order
        self.h_C = np.ascontiguousarray(np.transpose(padded_C), dtype=np.float64)

        # Build OpenCL data structures
            # Read only
                # Brownian motion
        self.d_noises = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_noises)
        self.d_K = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_K)
        self.d_C = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_C)
            # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_un)
        # might not need these
        self.d_udn_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn_x)
        self.d_udn_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn_y)
        self.d_udn_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn_z)
        # NOTE we must use COPY_HOST_PTR here
        self.d_udn1_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn1_x)
        self.d_udn1_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn1_y)
        self.d_udn1_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn1_z)

        self.d_bdn1_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_bdn1_x)
        self.d_bdn1_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_bdn1_y)
        self.d_bdn1_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_bdn1_z)
            # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_damage)

    def reset_sample(self, model, steps):
        # Displacements
        self.h_un = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Reset initial brownian forcing and forces to 0
        self.h_udn_x = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn_y = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn_z = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1_x = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1_y = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1_z = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_bdn1_x = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_bdn1_y = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_bdn1_z = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.zeros(model.nnodes).astype(np.float64)

        # Horizons
        self.h_horizons = model.horizons

        # Gaussian noise vectors
        self.h_noises = self.noise(model.nnodes, steps)

        # Build OpenCL data structures
            # Read only
                # Brownian motion
        self.d_noises = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_noises)
            # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_un)
        # NOTE we must use COPY_HOST_PTR here
        self.d_udn_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn_x)
        self.d_udn_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn_y)
        self.d_udn_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn_z)
        self.d_udn1_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn1_x)
        self.d_udn1_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn1_y)
        self.d_udn1_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_udn1_z)

        self.d_bdn1_x = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_bdn1_x)
        self.d_bdn1_y = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_bdn1_y)
        self.d_bdn1_z = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_bdn1_z)
            # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_damage)

    def runtime(self, model, step):
        # Time marching Part 1
        self.cl_kernel_update_displacement(self.queue, (model.nnodes,), None, 
                                           self.d_udn1_x,
                                           self.d_udn1_y,
                                           self.d_udn1_z,
                                           self.d_un,
                                           self.d_bdn1_x,
                                           self.d_bdn1_y,
                                           self.d_bdn1_z,
                                           self.d_bdn_x,
                                           self.d_bdn_y,
                                           self.d_bdn_z,
                                           self.d_noises,
                                           self.d_bc_types,
                                           self.d_bc_values,
                                           np.intc(step),
                                           self.h_displacement_load_scale)
        # Time marching Part 2: Calc forces
        self.cl_kernel_update_acceleration(self.queue, (model.nnodes * model.max_horizon_length,), (model.max_horizon_length,),
                                           self.d_un,
                                           self.d_udn_x,
                                           self.d_udn_y,
                                           self.d_udn_z,
                                           self.d_vols, 
                                           self.d_horizons, 
                                           self.d_coords, 
                                           self.d_bond_stiffness,
                                           self.d_bond_critical_stretch,
                                           self.d_force_bc_types,
                                           self.d_force_bc_values,
                                           self.local_mem_x,
                                           self.local_mem_y,
                                           self.local_mem_z,
                                           self.h_bond_stiffness_const,
                                           self.h_bond_critical_stretch_const,
                                           self.h_force_load_scale,
                                           self.h_displacement_load_scale
                                           )

        # Covariance matrix multiplication of forces
        self.cl_kernel_matrix_vector_mul2(self.queue, (self.h_m,self.h_p), (self.h_mdash, self.h_p),
                            self.d_K, self.d_udn_x, self.d_udn1_x, self.local_mem_mvmul2_1, self.h_m, self.h_n)
        self.cl_kernel_matrix_vector_mul2(self.queue, (self.h_m,self.h_p), (self.h_mdash, self.h_p),
                            self.d_K, self.d_udn_y, self.d_udn1_y, self.local_mem_mvmul2_2, self.h_m, self.h_n)
        self.cl_kernel_matrix_vector_mul2(self.queue, (self.h_m,self.h_p), (self.h_mdash, self.h_p),
                            self.d_K, self.d_udn_z, self.d_udn1_z, self.local_mem_mvmul2_3, self.h_m, self.h_n)

        # Sampling of correlated Brownian noise
        self.cl_kernel_matrix_vector_mul2(self.queue, (self.h_m,self.h_p), (self.h_mdash, self.h_p),
                            self.d_C, self.d_bdn_x, self.d_bdn1_x, self.local_mem_mvmul2_4, self.h_m, self.h_n)
        self.cl_kernel_matrix_vector_mul2(self.queue, (self.h_m,self.h_p), (self.h_mdash, self.h_p),
                            self.d_C, self.d_bdn_y, self.d_bdn1_y, self.local_mem_mvmul2_5, self.h_m, self.h_n)
        self.cl_kernel_matrix_vector_mul2(self.queue, (self.h_m,self.h_p), (self.h_mdash, self.h_p),
                            self.d_C, self.d_udn_z, self.d_udn1_z, self.local_mem_mvmul2_6, self.h_m, self.h_n)
        
    def write(self, model, t, sample, realisation):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_reduce_damage(self.queue, (model.nnodes * model.max_horizon_length,),
                                  (model.max_horizon_length,), self.d_horizons,
                                           self.d_horizons_lengths, self.d_damage, self.local_mem)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        vtk.write("output/U_"+"sample" + str(sample) +"realisation" +str(realisation) + "t" + str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        vtk.writeDamage("output/damage_" + "sample" + str(sample)+ "realisation" +str(realisation) + ".vtk", "Title", self.h_damage)
        return self.h_damage, self.h_un
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.h_force_load_scale = np.float64(load_scale)
    def incrementDisplacement(self, model, displacement_scale):
        # update the host force load scale
        self.h_displacement_load_scale = np.float64(displacement_scale)



def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units, \n")
            sys.stdout.write("a max of ")
            sys.stdout.write(str(device_id.max_work_group_size))
            sys.stdout.write(" work-items per work-group, \n")
            sys.stdout.write("a max work item dimensions of ")
            sys.stdout.write(str(device_id.max_work_item_dimensions))
            sys.stdout.write(", \na max work item sizes of ")
            sys.stdout.write(str(device_id.max_work_item_sizes))
            sys.stdout.write(",\nand device local memory size is ")
            sys.stdout.write(str(device_id.local_mem_size))
            sys.stdout.write(" bytes. \n")
            sys.stdout.flush()