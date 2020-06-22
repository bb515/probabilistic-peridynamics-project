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
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import EulerOpenCLOptimised
from pstats import SortKey, Stats
#import matplotlib.pyplot as plt
import time
import shutil
import os
# TODO: check that BC are correct on this main file

mesh_file_name = '1000beam3DT.msh'
mesh_file = pathlib.Path(__file__).parent.absolute() / mesh_file_name

@initial_crack_helper
def is_crack(x, y):
    output = 0
    return output

def is_tip(horizon, x):
    output = 0
    if mesh_file_name == '1000beam3DT.msh':
        if x[0] > 1.0 - 1./3 * horizon:
            output = 1
    return output

def is_rebar(p):
    """ Function to determine whether the node coordinate is rebar
    """
    p = p[1:] # y and z coordinates for this node
    if mesh_file_name == '1000beam3DT.msh':
        # Beam type 1 for flexural failure beam
        # Beam type 2 for shear failure beam
        beam_type = 2
        if beam_type == 1:
            bar_centers = [
                    # Tensile bars 25mm of cover, WARNING: only gives 21.8mm inner spacing of bars
                    np.array((0.0321, 0.185)),
                    np.array((0.0679, 0.185))]
            rad_t = 0.00705236
            
            radii = [
                    rad_t,
                    rad_t]
            costs = [ np.sum(np.square(cent - p) - (np.square(rad))) for cent, rad in zip(bar_centers, radii) ]
            if any( c <= 0 for c in costs ):
                return True
            else:
                return False
        elif beam_type ==2:
            bar_centers = [
                    # Tensile bars 25mm of cover, WARNING: only gives 7.6mm inner spacing of bars
                    np.array((0.0356, 0.185)),
                    np.array((0.0644, 0.185))]
            rad_t = 0.0105786
            
            radii = [
                    rad_t,
                    rad_t]
            costs = [ np.sum(np.square(cent - p) - (np.square(rad))) for cent, rad in zip(bar_centers, radii) ]
            if any( c <= 0 for c in costs ):
                return True
            else:
                return False
    else:
        return False

def bond_type(x, y):
    """ 
    Determines bond type given pair of node coordinates.
    Usage:
        'plain = 1' will return a plain concrete bond for all bonds, an so a
    plain concrete beam.
        'plain = 0' will return a concrete beam with some rebar as specified
        in "is_rebar()"
    """
    plain = 1
    output = 0 # default to concrete
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
    if mesh_file_name == '1000beam3DT.msh':
        bnd = [2, 2, 2]
        if x[0] < 1.5 * horizon:
            bnd[0] = 0
            bnd[1] = 0
            bnd[2] = 0
        if x[0] > 1.0 - 0.2* horizon:
            bnd[2] = 1
    return bnd

def is_forces_boundary(horizon, x):
    """
    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    if mesh_file_name == '1000beam3DT.msh':
        bnd = [2, 2, 2]
        if x[0] > 1.0 - 0.2 * horizon:
            bnd[2] = -1
    return bnd

def boundary_function(model):
    """ 
    Initiates displacement boundary conditions,
    also define the 'tip' (for plotting displacements)
    """
    load_rate = 1e-8
    #theta = 18.75
    # initiate
    model.bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
    model.tip_types = np.zeros(model.nnodes, dtype=np.intc)

    # Find the boundary nodes and apply the displacement values
    for i in range(0, model.nnodes):
        # Define boundary types and values
        bnd = is_boundary(model.horizon, model.coords[i][:])
        model.bc_types[i, 0] = np.intc(bnd[0])
        model.bc_types[i, 1] = np.intc(bnd[1])
        model.bc_types[i, 2] = np.intc((bnd[2]))
        model.bc_values[i, 0] = np.float64(bnd[0] * 0.5 * load_rate)
        model.bc_values[i, 1] = np.float64(bnd[1] * 0.5 * load_rate)
        model.bc_values[i, 2] = np.float64(bnd[2] * 0.5 * load_rate)
        # Define tip here
        tip = is_tip(model.horizon, model.coords[i][:])
        model.tip_types[i] = np.intc(tip)
    print(np.max(model.tip_types), 'max_tip_types')
    
def boundary_forces_function(model):
    """ 
    Initiates boundary forces. The units are force per unit volume.
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
    print('number of force BC nodes', num_force_bc_nodes)
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

    volume_total = 1.0 * 0.1 * 0.2
    density_concrete = 2400
    self_weight = 1.*density_concrete * volume_total * 9.81
    youngs_modulus_concrete = 1.*22e9
    youngs_modulus_steel = 1.*210e9
    #tensile_strength_concrete = 2.6e6
    # Set simulation parameters
    # Two materials in this example, that is 'concrete' and 'steel'
    dx = np.power(1.*volume_total/67500,1./3)
    horizon = dx * np.pi 
    damping = 2.0e6 # damping term
    # Peridynamic bond stiffness, c
    poisson_ratio = 0.25
    bulk_modulus_concrete = youngs_modulus_concrete/ (3* (1 - 2*poisson_ratio))
    bulk_modulus_steel = youngs_modulus_steel / (3* (1 - 2*poisson_ratio))
    bond_stiffness_concrete = (
    np.double((18.00 * bulk_modulus_concrete) /
    (np.pi * np.power(horizon, 4)))
    )
    bond_stiffness_steel = (
    np.double((18.00 * bulk_modulus_steel) /
    (np.pi * np.power(horizon, 4)))
    )
    critical_strain_concrete = np.double(0.000533) # check this value
    #critical_strain_concrete = np.double(1.0) # bond breakage off
    critical_strain_steel = np.double(0.01)
    crack_length = np.double(0)
    model = OpenCL(mesh_file_name,
                   density = density_concrete,
                   horizon = horizon,
                   damping = damping,
                   bond_stiffness_concrete = bond_stiffness_concrete, 
                   bond_stiffness_steel = bond_stiffness_steel,
                   critical_strain_concrete = critical_strain_concrete,
                   critical_strain_steel = critical_strain_steel,
                   crack_length = crack_length,
                   volume_total=volume_total,
                   bond_type=bond_type,
                   network_file_name = 'Network.vtk',
                   initial_crack=[],
                   dimensions=3,
                   transfinite=0,
                   precise_stiffness_correction=1)
    
    #saf_fac = 0.2 # Typical values 0.70 to 0.95 (Sandia PeridynamicSoftwareRoadmap)
    #model.dt = (
    # 0.8 * np.power( 2.0 * density_concrete * dx / 
    # (np.pi * np.power(model.horizon, 2.0) * dx * model.bond_stiffness_concrete), 0.5)
    # * saf_fac
    # )
    model.dt = 5.7e-14
    model.max_reaction = 1.* self_weight # in newtons, about 85 times self weight
    model.load_scale_rate = 1/100

    # Set force and displacement boundary conditions
    boundary_function(model)
    boundary_forces_function(model)

    integrator = EulerOpenCLOptimised(model)

    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')

    damage_data, damage_sum_data, tip_displacement_data = model.simulate(model, sample=1, steps=2000, integrator=integrator, write=2000, toolbar=0)
# =============================================================================
#     plt.figure(1)
#     plt.title('damage over time')
#     plt.plot(damage_data)
#     plt.figure(2)
#     plt.title('tip displacement over time')
#     plt.plot(tip_displacement_data)
#     plt.show()
# =============================================================================
    print(damage_sum_data)
    print('TOTAL TIME REQUIRED {}'.format(time.time() - st))
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()