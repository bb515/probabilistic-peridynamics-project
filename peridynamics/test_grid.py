# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:26:39 2020

@author: Ben Boys
"""

import numpy as np
import grid as fem
import matplotlib.pyplot as plt
# test grid.py
dim = 2
nnodes = 200
L = [1 , 1]
n = [10, 10]
hf = np.ones(dim)
order = 1
X0 = [0.0, 0.0] # bottom left
myGrid = fem.Grid()
myGrid.build_structured_mesh(L,n,X0)
particle_coords = np.random.rand(nnodes, dim)
for i in range(0, nnodes):
    particle_coords[i][0] *= L[0]
    particle_coords[i][1] *= L[1]
local_coords, p2e = myGrid.particle_to_cell(particle_coords)
print(local_coords.shape, 'shape')

coords_array = np.array(local_coords)
print(np.shape(coords_array), 'shape2')
coords_arrayT = np.transpose(coords_array)
print(np.shape(coords_arrayT), 'shape3')
plt.scatter(coords_arrayT[0], coords_arrayT[1])
plt.show()