////////////////////////////////////////////////////////////////////////////////
//
// opencl_euler_stochastic_optimised.cl (each work item does one bond, but each work group does one node)
//
// OpenCL Peridynamics kernels for Euler integrator
//
// Based on code from Copyright (c) Farshid Mossaiby, 2016, 2017. Adapted for python.
//
////////////////////////////////////////////////////////////////////////////////

// Includes, project
#include "opencl_enable_fp64.cl"
// Macros
#define DPN 3
// MAX_HORIZON_LENGTH, GLOBAL_DIMENSION, LOCAL_DIMENSION, PD_DT, PD_E, PD_S0, PD_NODE_NO, PD_DPN_NODE_NO, PD_DPN_NODE_NO_ROUNDED will be defined on JIT compiler's command line

// Update displacements
__kernel void
	UpdateDisplacement(
        __global double const * Udn1_x,
        __global double const * Udn1_y,
        __global double const * Udn1_z,
        __global double * Un,
		__global double * const Bdn1_x,
        __global double * const Bdn1_y,
        __global double * const Bdn1_z,
        __global double * Bdn_x,
        __global double * Bdn_y,
        __global double * Bdn_z,
        __global double * const Noises,
		__global int const * BCTypes,
		__global double const * BCValues,
		int step,
        double DISPLACEMENT_LOAD_SCALE
	)
{
	const int i = get_global_id(0);

	if (i < PD_NODE_NO)
	{
        // Update acceleration with Brownian forcing
		Un[DPN*i+0] = (BCTypes[DPN*i+0] == 2 ? (Un[DPN*i+0] + Bdn1_x[i] + PD_DT * Udn1_x[i]): (Un[DPN*i+0] + DISPLACEMENT_LOAD_SCALE * BCValues[DPN*i+0]));
        Un[DPN*i+1] = (BCTypes[DPN*i+1] == 2 ? (Un[DPN*i+1] + Bdn1_y[i] + PD_DT * Udn1_y[i]): (Un[DPN*i+1] + DISPLACEMENT_LOAD_SCALE * BCValues[DPN*i+1]));
        Un[DPN*i+2] = (BCTypes[DPN*i+2] == 2 ? (Un[DPN*i+2] + Bdn1_z[i] + PD_DT * Udn1_z[i]): (Un[DPN*i+2] + DISPLACEMENT_LOAD_SCALE * BCValues[DPN*i+2]));
        // Load new Brownian noise into global memory
        Bdn_x[i] = Noises[step * PD_DPN_NODE_NO + DPN*i + 0];
        Bdn_y[i] = Noises[step * PD_DPN_NODE_NO + DPN*i + 1];
        Bdn_z[i] = Noises[step * PD_DPN_NODE_NO + DPN*i + 2]; 
    }
}

// Time Integration step
__kernel void
	UpdateAcceleration(
    __global double const * Un,
    __global double * Udn_x,
    __global double * Udn_y,
    __global double * Udn_z,
    __global double const * Vols,
	__global int * Horizons,
	__global double const * Nodes,
	__global double const * Stiffnesses,
	__global double const * FailStretches,
    __global int const * FCTypes,
    __global double const * FCValues,
    __local double * local_cache_x,
    __local double * local_cache_y,
    __local double * local_cache_z,
    double PD_STIFFNESS,
    double PD_STRETCH,
    double FORCE_LOAD_SCALE,
    double DISPLACEMENT_LOAD_SCALE
	)
{
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, local_size] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the MAX_HORIZONS_LENGTHS, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);

	if ((global_id < (PD_NODE_NO * local_size)) && (local_id >= 0) && (local_id < local_size))
    {
        // Find corresponding node id
        const double temp = global_id / local_size;
        const int node_id_i = floor(temp);

        // Access local node within node_id_i's horizon with corresponding node_id_j,
        const int node_id_j = Horizons[global_id];
        
        if (node_id_j != -1) // If bond is not broken
        {
        const double xi_x = Nodes[DPN * node_id_j + 0] - Nodes[DPN * node_id_i + 0];
        const double xi_y = Nodes[DPN * node_id_j + 1] - Nodes[DPN * node_id_i + 1];
        const double xi_z = Nodes[DPN * node_id_j + 2] - Nodes[DPN * node_id_i + 2];

        const double xi_eta_x = Un[DPN * node_id_j + 0] - Un[DPN * node_id_i + 0] + xi_x;
        const double xi_eta_y = Un[DPN * node_id_j + 1] - Un[DPN * node_id_i + 1] + xi_y;
        const double xi_eta_z = Un[DPN * node_id_j + 2] - Un[DPN * node_id_i + 2] + xi_z;

        // Could instead here approximate bond stretch with a Taylor expansion, avoiding costly sqrt()
        const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
        const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
        const double y_xi = (y - xi);

        const double cx = xi_eta_x / y;
        const double cy = xi_eta_y / y;
        const double cz = xi_eta_z / y;

        const double _E = PD_STIFFNESS * Stiffnesses[global_id];
        const double _A = Vols[node_id_j];
        const double _L = xi;

        const double _EAL = _E * _A / _L;

        // Copy bond forces into local memory
        local_cache_x[local_id] = _EAL * cx * y_xi;
        local_cache_y[local_id] = _EAL * cy * y_xi;
        local_cache_z[local_id] = _EAL * cz * y_xi;

        // Check for state of bonds here
        const double PD_S0 = PD_STRETCH * FailStretches[global_id];
        const double s = (y - xi) / xi;
        if (s > PD_S0)
        {
            Horizons[global_id] = -1;  // Break the bond
        }
    }
    else // bond is broken
    {
        local_cache_x[local_id] = 0.00;
        local_cache_y[local_id] = 0.00;
        local_cache_z[local_id] = 0.00;
    }

    // Wait for all threads to catch up
    barrier(CLK_LOCAL_MEM_FENCE);
    // Parallel reduction of the bond force onto node force
    for (int i = local_size/2; i > 0; i /= 2){
        if(local_id < i){
            local_cache_x[local_id] += local_cache_x[local_id + i];
            local_cache_y[local_id] += local_cache_y[local_id + i];
            local_cache_z[local_id] += local_cache_z[local_id + i];
        }
        //Wait for all threads to catch up 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        //Get the reduced forces
        // node_no == node_id_i
        int node_no = global_id/local_size;
        // Update accelerations in each direction
        Udn_x[node_no] = (FCTypes[DPN * node_no + 0] == 2 ? local_cache_x[0] : (local_cache_x[0] + FORCE_LOAD_SCALE * FCValues[DPN * node_no + 0]));
        Udn_y[node_no] = (FCTypes[DPN * node_no + 1] == 2 ? local_cache_y[0] : (local_cache_y[0] + FORCE_LOAD_SCALE * FCValues[DPN * node_no + 1]));
        Udn_z[node_no] = (FCTypes[DPN * node_no + 2] == 2 ? local_cache_y[0] : (local_cache_y[0] + FORCE_LOAD_SCALE * FCValues[DPN * node_no + 2]));
    }
  }
}

__kernel void 
    ReduceDamage(
        __global int const *Horizons,
		__global int const *HorizonLengths,
        __global double *Phi,
        __local double* local_cache
    )
{
    int global_id = get_global_id(0); 
    int local_id = get_local_id(0); 
    // local size is the MAX_HORIZONS_LENGTHS and must be a power of 2
    int local_size = get_local_size(0); 
    
    //Copy values into local memory 
    local_cache[local_id] = Horizons[global_id] != -1 ? 1.00 : 0.00; 

    //Wait for all threads to catch up 
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_size/2; i > 0; i /= 2){
        if(local_id < i){
            local_cache[local_id] += local_cache[local_id + i];
        } 
        //Wait for all threads to catch up 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        //Get the reduced forces
        int node_id = global_id/local_size;
        // Update damage
        Phi[node_id] = 1.00 - (double) local_cache[0] / (double) (HorizonLengths[node_id]);
    }
}

// Matrix vector multiplication
// One thread per dot product
__kernel void gemv1(__global const double * a,__global const double * x, 
                    __global double * y,int m,int n)
{
  double sum = 0.0;
  int i = get_global_id(0); // row index
  for (int k=0;k<n;k++)
    {
      sum += a[i + m*k] * x[k];
    }
  y[i] = sum;
}


// P threads per dot product
#define ROW_DIM 0
#define COL_DIM 1
// P threads per row compute 1/P-th of each dot product.
// WORK has P columns and get_local_size(0) rows.
__kernel void gemv2(__global const double * a,
                    __global const double * x,
		            __global double * y,
		            __local double * work,
		            int m,int n)
{
  // Compute partial dot product
  double sum = (double)0;
  for (int k=get_global_id(COL_DIM);k<n;k+=get_global_size(COL_DIM))
    {
      sum += a[get_global_id(ROW_DIM)+m*k] * x[k];
    }

  // Each thread stores its partial sum in WORK
  int rows = get_local_size(ROW_DIM); // rows in group
  int cols = get_local_size(COL_DIM); // initial cols in group
  int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
  int jj = get_local_id(COL_DIM); // block index in column, 0<=jj<cols
  work[ii+rows*jj] = sum;
  barrier(CLK_LOCAL_MEM_FENCE); // sync group

  // Reduce sums in log2(cols) steps
  while ( cols > 1 )
    {
      cols >>= 1;
      if (jj < cols) work[ii+rows*jj] += work[ii+rows*(jj+cols)];
      barrier(CLK_LOCAL_MEM_FENCE); // sync group
    }

  // Write final result in Y
  if ( jj == 0 ) y[get_global_id(ROW_DIM)] = work[ii];
}



// P threads per row compute 1/P-th of each dot product.
// WORK has N/P entries.
__kernel void gemv3(__global const double * a,__global const double * x,
		    __global double * y,
		    __local double * work,
		    int m,int n)
{
  // Load a slice of X in WORK, using all available threads
  int ncols = n / get_global_size(COL_DIM); // nb values to load
  int col0 = ncols * get_global_id(COL_DIM); // first value to load
  for (int k=0;k<ncols;k+=get_local_size(ROW_DIM))
    {
      int col = k+get_local_id(ROW_DIM);
      if (col < ncols) work[col] = x[col0+col];
    }
  barrier(CLK_LOCAL_MEM_FENCE); // sync group

  // Compute partial dot product
  double sum = (double)0;
  for (int k=0;k<ncols;k++)
    {
      sum += a[get_global_id(ROW_DIM)+m*(col0+k)] * work[k];
    }

  // Store in Y (P columns per row)
  y[get_global_id(ROW_DIM)+m*get_global_id(COL_DIM)] = sum;
}

// Reduce M = get_global_size(0) rows of P values in matrix Y.
// Stores the result in first column of Y.
__kernel void reduce_rows(__global double * y,int m,int p)
{
  int row = get_global_id(0);
  double sum = (double)0;
  for (int col=0;col<p;col++) sum += y[row + m*col];
  y[row] = sum;
}