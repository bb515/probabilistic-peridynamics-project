"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import OpenCL
from peridynamics import OpenCLProbabilistic
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import EulerOpenCL
from peridynamics.integrators import EulerOpenCLOptimised
from peridynamics.integrators import EulerOpenCLOptimisedLumped2
from peridynamics.integrators import EulerOpenCLMCMC
from peridynamics.integrators import EulerStochasticOptimised
from peridynamics.integrators import RK4
from peridynamics.integrators import DormandPrinceOptimised
from peridynamics.post_processing import vtk
from pstats import SortKey, Stats
import matplotlib.pyplot as plt
import time
import shutil
import os

mesh_file_name = 'plate.msh'
mesh_file = pathlib.Path(__file__).parent.absolute() / mesh_file_name

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['COMPUTE_PROFILE'] = '1'
os.environ['PYOPENCL_CTX'] = '0:0'
@initial_crack_helper
def is_crack(x, y):
    output = 0
# =============================================================================
#     crack_length = 0.3
#     p1 = x
#     p2 = y
#     if x[0] > y[0]:
#         p2 = x
#         p1 = y
#     # 1e-6 makes it fall one side of central line of particles
#     if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
#         # draw a straight line between them
#         m = (p2[1] - p1[1]) / (p2[0] - p1[0])
#         c = p1[1] - m * p1[0]
#         # height a x = 0.5
#         height = m * 0.5 + c
#         if (height > 0.5 * (1 - crack_length)
#                 and height < 0.5 * (1 + crack_length)):
#             output = 1
# =============================================================================
    return output

def is_tip(horizon, x):
    output = 0
    if mesh_file_name == 'plate.msh':
        if x[0] > 1.0 - 1. * horizon:
            output = 1
    return output

def is_rebar(x):
    """ Function to determine whether the node coordinate is rebar
    """
    output = False
    if mesh_file_name == 'plate.msh':
        if x[0] < 0.05:
            output = True
        if x[0] > 0.1:
            output = True
    return output

def bond_type(x, y):
    """ 
    Determines bond type given pair of node coordinates.
    Usage:
        'plain = 1' will return a plain concrete bond for all bonds, an so a
    plain concrete beam.
        'plain = 0' will return a concrete beam with some rebar as specified
        in "is_rebar()"
    """
    plain = 0
    output = 'concrete' # default to concrete
    bool1 = is_rebar(x)
    bool2 = is_rebar(y)
    if plain == 1:
        output = 'concrete'
    elif bool1 and bool2:
        output = 'steel'
    elif bool1 != bool2:
        output = 'interface'
    else:
        output = 'concrete'
    return output

def is_boundary(horizon, x):
    """
    Function which marks displacement boundary constrained particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is displacement loaded IN -ve direction
    1 is displacement loaded IN +ve direction
    0 is clamped boundary
    """
    if mesh_file_name == 'plate.msh':
        # Does not live on a boundary
        bnd = 2
        # Does live on boundary
        if x[0] < 1.5 * horizon:
            bnd = -1
        elif x[0] > 0.15 - 1.5 * horizon:
            bnd = 1
    return bnd

def is_forces_boundary(horizon, x):
    """
    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    if mesh_file_name == 'plate.msh':
        bnd = [2, 2, 2]
        #if x[0] > 1.0 - 1.5 * horizon:
            #bnd[2] = 1
    return bnd

def boundary_function(model, displacement_rate):
    """ 
    Initiates displacement boundary conditions,
    also define the 'tip' (for plotting displacements)
    """
    #initiate containers
    model.bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
    model.tip_types = np.zeros(model.nnodes, dtype=np.intc)

    # Find the boundary nodes and apply the displacement values
    for i in range(0, model.nnodes):
        # Define boundary types and values
        bnd = is_boundary(model.horizon, model.coords[i][:])
        model.bc_types[i, 0] = np.intc((bnd))
        model.bc_types[i, 1] = np.intc((bnd))
        model.bc_types[i, 2] = np.intc((bnd))
        model.bc_values[i, 0] = np.float64(bnd * displacement_rate)

        # Define tip here
        tip = is_tip(model.horizon, model.coords[i][:])
        model.tip_types[i] = np.intc(tip)

def boundary_forces_function(model):
    """ 
    Initiates boundary forces
    """
    model.force_bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.force_bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

    # Find the force boundary nodes and find amount of boundary nodes
    num_force_bc_nodes = 0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.horizon, model.coords[i][:])
        if -1 in bnd:
            num_force_bc_nodes += 1
        elif 1 in bnd:
            num_force_bc_nodes += 1
        model.force_bc_types[i, 0] = np.intc((bnd[0]))
        model.force_bc_types[i, 1] = np.intc((bnd[1]))
        model.force_bc_types[i, 2] = np.intc((bnd[2]))

    model.num_force_bc_nodes = num_force_bc_nodes
                
def main():
    """
    3D canteliver beam peridynamics simulation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    st = time.time()
    bulk_modulus_concrete = 0.04
    critical_strain_concrete = 0.005
    horizon = np.pi*1.7e-3
    
    # Set simulation parameters
# =============================================================================
#     model = OpenCL(mesh_file_name, 
#                density = 1.0, 
#                horizon = horizon,
#                damping = 1.0,
#                bond_stiffness_concrete = (
#                        np.double((18.00 * bulk_modulus_concrete) /
#                                  (np.pi * np.power(horizon, 4)))
#                        ),
#                bond_stiffness_steel = (
#                np.double((18.00 * bulk_modulus_concrete) /
#                          (np.pi * np.power(horizon, 4)))
#                ),
#                critical_strain_concrete = critical_strain_concrete, # was 0.005
#                critical_strain_steel = 1.0,
#                crack_length = 0.0,
#                volume_total=(0.15*0.1 - np.pi *0.00148**2)*0.0017,
#                bond_type=bond_type,
#                network_file_name = 'Network_6.vtk',
#                initial_crack=[],
#                dimensions=3,
#                transfinite=0,
#                precise_stiffness_correction=1)
# =============================================================================

    model = OpenCLProbabilistic(mesh_file_name, 
                                density = 1.0,
                                horizon = horizon, 
                                damping = 1.0,
                                dx = 0.01,
                                bond_stiffness_const = 1.0,
                                critical_stretch_const = 1.0,
                                sigma = np.exp(-4.0), 
                                l = np.exp(-5.0),
                                crack_length = 0.0,
                                volume_total=(0.15*0.1 - np.pi *0.00148**2)*0.0017,
                                bond_type=bond_type,
                                network_file_name = 'Network_6.vtk',
                                initial_crack=[],
                                dimensions=3,
                                transfinite= 0,
                                precise_stiffness_correction = 1)
    model.horizon= horizon
    # damping
    #model.dt = np.double(5.0e-5/1.1)
    model.dt = np.double(4.0e-5)
    # Set force and displacement boundary conditions
    displacement_rate = 5.0e-7
    boundary_function(model, displacement_rate)
    boundary_forces_function(model)
    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')
    model._set_D(0.9, 1.0)
    #integrator = EulerOpenCLMCMC(model)#, error_size_max = 1e-1, error_size_min = 1e-30)
    #integrator.reset(model)
    #damage_sum_data, tip_displacement_data, tip_shear_force_data = model.simulate(model, sample=1, steps=1000, integrator=integrator, write=1000, toolbar=0,
                                                                                  #displacement_rate = displacement_rate)
    integrator = EulerStochasticOptimised(model)
    integrator.reset(model, steps=1000)
    samples = 4
    mean_damage = np.zeros(model.nnodes)
    mean_displacement = np.zeros((model.nnodes, model.degrees_freedom))
    model._set_H(np.exp(-5.0), np.exp(-4.0), 1.0, 1.0)
    for sample in range(samples):
        integrator.reset_sample(model, 1000)
        damage_data, displacement_data= model.simulate(model, sample=sample, realisation=1, steps=1000, integrator=integrator, write=1000, toolbar=0, displacement_rate = displacement_rate)
        mean_damage += damage_data
        mean_displacement += displacement_data
    mean_damage = np.divide(mean_damage, samples)
    mean_displacement = np.divide(mean_displacement, samples)
# =============================================================================
#     vtk.write("output/U_"+"samples" + str(samples) +"mean" + "t" + str(1000) + ".vtk", "Solution time step = "+str(1000),
#                   model.coords, mean_damage, mean_displacement)
#     vtk.writeDamage("output/damage_" + "samples" + str(samples)+ "mean"+ ".vtk", "Title", mean_damage)
# =============================================================================
    
    #vtk.writeDamage("output/damage_" + "mcmc_6" + ".vtk", "Title", damage_data)
    print('TOTAL TIME REQUIRED {}'.format(time.time() - st))
    plt.figure(1)
    plt.title('damage over time')
    plt.plot(damage_sum_data)
    #plt.figure(2)
    #plt.title('tip displacement over time')
    #plt.plot(tip_displacement_data)
    #plt.show()
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())

if __name__ == "__main__":
    main()