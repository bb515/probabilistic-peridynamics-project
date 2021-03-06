"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import OpenCLProbabilistic
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import EulerStochastic
from pstats import SortKey, Stats
#import scipy.stats as sp
import shutil
import os
import mcmc
import csv
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
mesh_file_name = 'test.msh'
mesh_file = pathlib.Path(__file__).parent.absolute() / mesh_file_name

@initial_crack_helper
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

def is_tip(horizon, x):
    output = 0
    return output

def is_rebar(p):
    """ Function to determine whether the node coordinate is rebar
    """
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
    # Does not live on a boundary
    bnd = 2
    # Does live on boundary
    if x[0] < 1.5 * horizon:
        bnd = -1
    elif x[0] > 1.0 - 1.5 * horizon:
        bnd = 1
    return bnd

def is_forces_boundary(horizon, x):
    """
    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    bnd = 2
    return bnd

def boundary_function(model):
    """ 
    Initiates displacement boundary conditions,
    also define the 'tip' (for plotting displacements)
    """
    load_rate = 1e-5
    #theta = 18.75
    # initiate
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
        model.bc_values[i, 0] = np.float64(bnd * 0.5 * load_rate)
        #model.bc_values[i, 0] = np.float64(bnd * -0.5/theta * load_rate)

        # Define tip here
        tip = is_tip(model.horizon, model.coords[i][:])
        model.tip_types[i] = np.intc(tip)
    print(np.max(model.tip_types), 'max_tip_types')

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
        if bnd == -1:
            num_force_bc_nodes += 1
        elif bnd == 1:
            num_force_bc_nodes += 1
        model.force_bc_types[i, 0] = np.intc((bnd))
        model.force_bc_types[i, 1] = np.intc((bnd))
        model.force_bc_types[i, 2] = np.intc((bnd))

    model.num_force_bc_nodes = num_force_bc_nodes

    # Calculate initial forces
    model.force_bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
    load_scale = 0.0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.horizon, model.coords[i][:])
        if bnd == 1:
            pass
        elif bnd == -1:
            model.force_bc_values[i, 2] = np.float64(1.* bnd * model.max_reaction * load_scale / (model.num_force_bc_nodes))
def read_data(model):
    """
    Function for reading the damage data from the vtk file,
    damage_data and storing it in an array
    """
    def find_string(string, iline):
        """
        Finds position of a given string in vtk file.
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
    damage_data = []
    f = open("damage_data.vtk", "r")
    if f.mode == "r":
        iline = 0
        # Read the Max horizons length first
        row_as_list, iline = find_string('DAMAGE', iline)
        for i in range(0, model.nnodes):
            iline += 1
            line = f.readline()
            damage_data.append(np.float(line.split()[0]))
    damage_data= np.array(damage_data)
    return damage_data

def read_csv():
    data = [[],[],[]]
    data_smooth = [[],[],[]]
    with open(pathlib.Path(__file__).parent.absolute() / "nll.csv", 'r', newline='') as myfile:
        reader = csv.reader(myfile)
        for row in reader:
            data[0].append(eval(row[0]))
            data[1].append(eval(row[1]))
            data_ps = [float(i) for i in row[2:]]
            data[2].append(data_ps)
            data_smooth[0].append(eval(row[0]))
            data_smooth[1].append(eval(row[1]))
            data_smooth[2].append(np.mean(data_ps))
    return data, data_smooth

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

    volume_total = 1.0
    density_concrete = 1
    self_weight = 1.*density_concrete * volume_total * 9.81
    model = OpenCLProbabilistic(mesh_file_name, volume_total, sigma= np.exp(-7.0), l = np.exp(-4.5), bond_type=bond_type, initial_crack=is_crack)
    #dx = np.power(1.*volume_total/model.nnodes,1./(model.dimensions))
    # Set simulation parameters
    # not a transfinite mesh
    model.transfinite = 0
    # do precise stiffness correction factors
    model.precise_stiffness_correction = 1
    # Only one material in this example, that is 'concrete'
    model.density = density_concrete
    #self.horizon = dx * np.pi 
    model.horizon = 0.1
    model.family_volume = np.pi * np.power(model.horizon, 2)
    model.damping = 1 # damping term
    # Peridynamic bond stiffness, c
    model.bond_stiffness_concrete = (
            np.double((18.00 * 0.05) /
            (np.pi * np.power(model.horizon, 4)))
            )
    model.critical_strain_concrete = 0.005
    model.crackLength = np.double(0.3)
    model.dt = np.double(1e-3)
    model.max_reaction = 1.* self_weight
    model.load_scale_rate = 1
    # Set force and displacement boundary conditions
    boundary_function(model)
    boundary_forces_function(model)
    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')
    
    realisations = 5
    integrator = EulerStochastic(model)
    integrator.reset(model, steps=350)
    params = [[-10.5,-1.0]]
    samples = len(params)
    for sample in range(samples):
        # update (l, sigma)
        model._set_H(np.exp(params[sample][1]), np.exp(params[sample][0]))
        for realisation in range(realisations):
            integrator.reset(model, steps=350)
            sample_data = model.simulate(model, sample, realisation, steps=350, integrator=integrator, write=350, toolbar=0)
            print(np.sum(sample_data), 'sum of damage, realisation #', realisation)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()