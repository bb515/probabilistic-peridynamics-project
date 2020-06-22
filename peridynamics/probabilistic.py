# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:44:27 2020

@author: Ben Boys

Probabilstic functions for probabilistic peridynamics
"""
import numpy as np
import scipy.stats as sp

def multivar_normal(L, num_nodes):
    """ Fn for taking a single multivar normal sample covariance matrix with Cholesky factor, L
    """
    zeta = np.random.normal(0, 1, size = num_nodes)
    zeta = np.transpose(zeta)
    w_tild = np.dot(L, zeta) #vector
    return w_tild

def noise(C, K, num_nodes, num_steps, degrees_freedom = 3):
    """Takes sample from multivariate normal distribution 
    with covariance matrix whith Cholesky factor, L
    :arg L: Cholesky factor, C
    :arg C: Covariance matrix, K
    :arg samples: The number of degrees of freedom (read: dimensions) the
    noise is generated in, degault 3 i.e. x,y and z directions.
    :returns: num_nodes * 3 * num_steps array of noise
    :rtype: np.array dtype=float64
    """
    noise = []
    for i in range(degrees_freedom * num_steps):
        noise.append(multivar_normal(C, num_nodes))
    #brownian_motion = np.transpose(noise)
    #M = np.sqrt(np.multiply(K, 2))
    #noise_vector = np.linalg.dot(M, noise)
    return np.ascontiguousarray(noise, dtype=np.float64)

def brownian_noise(C, K, num_nodes, degrees_freedom = 3):
    """
    TODO
    """