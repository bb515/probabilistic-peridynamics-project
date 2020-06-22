# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:28:47 2020

@author: Ben Boys

pymc3
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from example_NLML import Model
from itertools import product

# Set the random field parameters
# random field resolution? Would this be the grid spacing.
field_stdev=9e-4
lamb=1e-2
resolution = None
field_mean = None
# What is mkl?
mkl = 8

my_model = Model(resolution, field_mean, mkl, field_stdev, lamb)


def model_wrapper(theta, datapoints):
    my_model.solve(theta)
    return my_model.get_data(datapoints)

def my_loglik(theta, datapoints, data, sigma):
    output = model_wrapper(theta, datapoints)
    return -(0.5/sigma**2)*np.sum((output - data)**2)

my_model.solve()
my_model.plot(transform_field=True)

class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood

ndraws = 4000  # number of draws from the distribution
nburn = 1000   # number of "burn-in points" (which we'll discard)

# create our Op
logl = LogLike(my_loglik, data, datapoints, sigma)

# use PyMC3 to sampler from log-likelihood
with pm.Model():
    # uniform priors on parameters
    parameters = []
    for i in range(mkl):
        parameters.append(pm.Uniform('theta_' + str(i), lower=-3., upper=3.))

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable(parameters)

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

    trace = pm.sample(ndraws, step=pm.Metropolis(), tune=nburn, discard_tuned_samples=True)