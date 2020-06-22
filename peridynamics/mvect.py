# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:33:41 2020

@author: Ben Boys


Test matrix vector multiplication kernels
"""
import pyopencl as cl
import numpy as np
import sys
import pathlib

sys.path.insert(1, pathlib.Path(__file__).parent.absolute() / 'peridynamics/kernels')
import time

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

np.random.seed(69)
               
x = np.random.normal(0, 1, (513))
A = np.random.normal(0, 1, (513, 513))

print(A)
print(x.shape)
print(A.shape)

y = np.dot(A, x)

h_m = np.intc(
        1<<(len(x)-1).bit_length()
        )
h_n = np.intc(len(x))


shape = np.shape(A)
padded_A = np.zeros((h_m, h_n))
padded_A[:shape[0],:shape[1]] = A

print(y.shape)

 # Initializing OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)   

# Print out device info
output_device_info(context.devices[0])

# Build the OpenCL program from file
kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/mvmul.cl").read()


# Build the programs
#program = cl.Program(context, kernelsource).build([options_string])

program = cl.Program(context, kernelsource).build()

cl_kernel_matrix_vector_mul = program.gemv1

# Set initial values in host memory
# horizons and horizons lengths
h_x = np.ascontiguousarray(x, dtype=np.float64)
h_A = np.ascontiguousarray(np.transpose(padded_A), dtype=np.float64)
h_y = np.empty((h_n), dtype=np.float64)

print(h_n)
print(h_m)

 # Read only
d_x = cl.Buffer(context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=h_x)
d_A = cl.Buffer(context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=h_A)

# Write only
d_y= cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_y.nbytes)

cl_kernel_matrix_vector_mul.set_scalar_arg_dtypes(
            [None, None, None, None, None])

start = cl.enqueue_marker(queue)
# Calc bond forces
cl_kernel_matrix_vector_mul(queue, (h_m,), (128,),
                            d_A, d_x, d_y, h_m, h_n)
cl.enqueue_copy(queue, h_y, d_y)
finish = cl.enqueue_marker(queue)

print('Time taken for kernel was', (start.profile.end-finish.profile.start)*1e-9)


zeros = np.subtract(h_y, y)

print(np.max(zeros))


