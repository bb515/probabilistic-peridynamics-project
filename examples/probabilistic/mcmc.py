# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:16:27 2020

@author: Ben Boys
"""
import numpy as np
import scipy.stats as sp
from scipy.special import gamma
from matplotlib import pyplot as plt

def gamma_prior_pdf(x, alpha =1., beta=100.0):
    """ pdf of zeta rv with values of alpha and beta
    """
    NUM1 = pow(gamma(alpha), -1)
    NUM2 = pow(beta, alpha)
    EXP = x*(alpha) - beta*pow(np.e, x)
    
    return(NUM1*NUM2*pow(np.e, EXP))

def trunc_normal_prior_pdf(x, mean, sigma):
    """ pdf of zeta rv with values of alpha and beta
    """
    if x < 0:
        return 0.0
    else:
        NUM1 = pow((x - mean), 2)
        NUM2 = -2* pow(sigma, 2)
        return(pow(np.e, (NUM1/NUM2)))

def get_log_likelihood(damage_data, model_sample):
    # Assume idependent, identically distributed with a variance of 1
    error = np.subtract(model_sample, damage_data)
    l2 = np.dot(error, error)
    nll = 1./2 * l2
    print(nll)
    likelihood = np.exp(-1.*nll)
    print(likelihood)
    return nll

def get_likelihood(damage_data, model_sample):
    # Assume idependent, identically distributed with a variance of 1
    error = np.subtract(model_sample, damage_data)
    l2 = np.dot(error, error)
    nll = 1./2 * l2
    likelihood = np.exp(-1.*nll)
    print(likelihood)
    return likelihood

# =============================================================================
# def get_KL_divergence(damage_data, model_sample):
#     # Taking the damage_data as ground truth
# =============================================================================

def beta_likelihood(damage_data, model_sample):
    mode = model_sample
    alpha = 2
    beta = 1/mode
    log_likelihood = 0
    for i in range(len(damage_data)):
        likelihood = sp.beta.pdf(damage_data, alpha, beta, loc = model_sample)
        log_likelihood += np.log(likelihood)
    total_likelihood = np.exp(log_likelihood)
    return total_likelihood


# =============================================================================
# x = np.linspace(-10.0, -2.0, 50)
# 
# y = gamma_prior_pdf(x)
# 
# plt.plot(x,y)
# =============================================================================
