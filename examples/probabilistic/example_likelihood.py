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
from peridynamics.integrators import EulerStochasticOptimised
from pstats import SortKey, Stats
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import scipy.stats as sp
import shutil
import os
import mcmc
import csv
#from matplotlib import pyplot as plt
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
        model.bc_values[i, 0] = np.float64(bnd * 0.5 * displacement_rate)

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

def read_csv(file):
    data = [[],[],[]]
    data_smooth = [[],[],[]]
    with open(pathlib.Path(__file__).parent.absolute() / file, 'r', newline='') as myfile:
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

    # Set simulation parameters
    model = OpenCLProbabilistic(mesh_file_name, 
                                density = 1.0,
                                horizon = 0.1, 
                                damping = 1.0,
                                dx = 0.01,
                                bond_stiffness_const = 1.0,
                                critical_stretch_const = 1.0,
                                sigma = np.exp(-3.8), 
                                l = np.exp(-4.8),
                                crack_length = 0.3,
                                volume_total=1.0,
                                bond_type=bond_type,
                                network_file_name = 'Network_2.vtk',
                                initial_crack=[],
                                dimensions=2,
                                transfinite= 0,
                                precise_stiffness_correction = 1)
    model.dt = np.double(1.2e-3)
    displacement_rate = 1e-5
    # Set force and displacement boundary conditions
    boundary_function(model, displacement_rate)
    boundary_forces_function(model)

    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')
    # Parameter grid search wrapper function
    # read the data
    damage_data = read_data(model)
    sigma_shape = 30
    lengths_shape = 30
    realisations = 40
    sigmas = np.linspace(-5.8, -3.8, sigma_shape)
    lengths = np.linspace(-4.8, 2.8, lengths_shape)
    samples = len(sigmas)*len(lengths)

    integrator = EulerStochasticOptimised(model)
    sample = 0

    # Evaluate the pdf of the distribution we want to sample from
    data = np.zeros((sigma_shape, lengths_shape))
    total_samples = 0

    with open(pathlib.Path(__file__).parent.absolute() / "likelihood.csv", 'w') as output:
                writer = csv.writer(output, lineterminator='\n')
    with open(pathlib.Path(__file__).parent.absolute() / "nll.csv", 'w') as output:
                writer = csv.writer(output, lineterminator='\n')
    for x in range(len(sigmas)):
        for y in range(len(lengths)):
            total_samples += 1
            print('Sample {}/{} Complete'.format(total_samples, samples))
            model._set_H(np.exp(lengths[y]), np.exp(sigmas[x]), bond_stiffness_const = 1.0, critical_stretch_const = 1.0)
            row = [sigmas[x], lengths[y]]
            row0 = [sigmas[x], lengths[y]]
            # Get the likelihood
            likelihood_sum = 0
            for realisation in range(realisations):
                integrator.reset(model, steps=350)
                sample_data = model.simulate(model, sample, realisation, steps=350, integrator=integrator, write=350, toolbar=0, displacement_rate = displacement_rate)
                print(np.sum(sample_data), 'sum of damage, realisation #', realisation)
                likelihood = mcmc.get_likelihood(damage_data, sample_data)
                likelihood_sum += likelihood
                row.append(likelihood)
            data[x,y] = -1.*np.log(likelihood_sum/realisations)
            with open(pathlib.Path(__file__).parent.absolute() / "nll.csv", 'a') as output:
                writer = csv.writer(output, lineterminator='\n')
                row0.append(data[x,y])
                writer.writerow(row0)
            with open(pathlib.Path(__file__).parent.absolute() / "likelihood.csv", 'a') as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerow(row)
# =============================================================================
#     data, data_smooth = read_csv('nll.csv')
#     
#     zs = np.array(data[2])
#     print(np.shape(zs))
#     r = np.shape(zs)[1]
#     print(r)
#     Zs = []
#     for i in range(r):
#         Zs.append(np.reshape(zs[:,i], (sigma_shape,lengths_shape)))
#         
#     z_flat = np.array(data_smooth[2])
#     
#     z = np.reshape(z_flat, (sigma_shape, lengths_shape))
#     print(z.shape)
#     
#     y = sigmas
#     x = lengths
#     
#     plt.contourf(x,y,z)
#     plt.plot(-1.00, -10.50, 'o', color = 'red')
#     plt.xlabel(r'Length scale, $l$')
#     plt.ylabel(r'Vertical noise scale, $\sigma$')
#     plt.show()
#     
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     X, Y = np.meshgrid(x, y)
#     surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#     
#     # Customize the z axis.
#     #ax.set_zlim(10, 20)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#     
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     
#     plt.show()
# =============================================================================

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()
