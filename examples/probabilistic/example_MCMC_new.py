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
from peridynamics.integrators import EulerOpenCLMCMC
from pstats import SortKey, Stats
import scipy.stats as sp
import shutil
import os
import mcmc
import csv
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
    model = OpenCL(mesh_file_name, 
                   density = 1.0,
                   horizon = 0.1, 
                   damping = 1.0,
                   dx = 0.01,
                   crack_length = 0.3,
                   volume_total=1.0,
                   bond_type=bond_type,
                   network_file_name = 'Network_2.vtk',
                   initial_crack=[],
                   dimensions=2,
                   transfinite= 0,
                   precise_stiffness_correction = 1)
    

    model.dt = np.double(1e-3)
    # Set force and displacement boundary conditions
    displacement_rate = 1e-5
    boundary_function(model, displacement_rate)
    boundary_forces_function(model)
    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')
    # MCMC wrapper function
    # read the data
    damage_data = read_data(model)
    samples = 10
    
    # Define start point of the Metropolis Hastings sampler w[1] is l, w[0] is sigma
    w_prev = [1.0, 1.0]
    
    # Define proposal density of the MCMC sampler
    w_cov = [[0.01, 0.0],[0.0, 0.01]]
    
    # Get the intial likelihood
    # update (l, sigma)
    model._set_D(w_prev[0], w_prev[1])
    integrator = EulerOpenCLMCMC(model)
    likelihood_prev = 0
    sample = 0
    realisation = 0
    integrator.reset(model)
    sample_data, tip_displacement_data, tip_shear_force_data = model.simulate(model, sample=1, steps=350, integrator=integrator, write=350, toolbar=0,
                                                                                  displacement_rate = displacement_rate)
    print(np.sum(sample_data), 'sum of damage, realisation #', realisation)
    likelihood_prev = mcmc.get_likelihood(damage_data, sample_data)
    assert likelihood_prev != 0, 'Floating point error on first likelihood value: likelihood must be more than 0'

    # Evaluate the pdf of the distribution we want to sample from
    prior_prev = mcmc.trunc_normal_prior_pdf(w_prev[0], 1.0, 3.0)*mcmc.trunc_normal_prior_pdf(w_prev[1], 1.0, 3.0)
    data = [[],[]]
    total_samples = 0
    
    for sample in range(samples):
        total_samples += 1
        print('Sample {}/{} Complete'.format(total_samples, samples))
        # Get proposal parameters
        w = sp.multivariate_normal.rvs(w_prev, w_cov, 1)
        # update (l, sigma)
        model._set_D(w[0], w[1])
        # Multiply two single variate prior distributions
        prior = mcmc.trunc_normal_prior_pdf(w[0], 1.0, 3.0)*mcmc.trunc_normal_prior_pdf(w[1], 1.0, 3.0)
        if prior ==0:
            None
        else:

            # Get the likelihood
            integrator.reset(model)
            sample_data, tip_displacement_data, tip_shear_force_data = model.simulate(model, sample=sample, steps=350, integrator=integrator, write=350, toolbar=0,
                                                                                  displacement_rate = displacement_rate)
            print(np.sum(sample_data), 'sum of damage, realisation #', realisation)
            likelihood = mcmc.get_likelihood(damage_data, sample_data)

            # compute acceptance ratio
            r = (prior * likelihood)/ (prior_prev * likelihood_prev)
            
            # Generate u from a unifrom distribution
            u = np.random.uniform()
            if u <= r:
                # accept the sample
                data[0].append(w[0])
                data[1].append(w[1])
                print('accepted sigma: ', w[0], 'accepted l: ', w[1])
                w_prev = w
                prior_prev = prior
                likelihood_prev = likelihood
                with open(pathlib.Path(__file__).parent.absolute() / "mcmc_current.csv", 'a', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow([w[0],w[1]])
            else:
                None
    
    # Perform the burn on the first 100 values
    burn = 0
    
    data[0] = data[0][burn:]
    data[1] = data[1][burn:]
    
    print(len(data[0]), 'LENGTH')
    mean = np.divide(np.sum(data, axis = 1), len(data[0]))
    print(mean, mean.shape, 'MEAN AND SHAPE')
        
    with open(pathlib.Path(__file__).parent.absolute() / "mean.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in mean:
            writer.writerow([val])
    if total_samples == 0:
        pass
    else:
        print('The percentage of accepted samples was {}%'.format(len(data[0])*100/(total_samples)))
    
    # Write data to a file
    with open(pathlib.Path(__file__).parent.absolute() / "mcmc.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        data_zipped = zip(*data)
        wr.writerow(data_zipped)
    
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()