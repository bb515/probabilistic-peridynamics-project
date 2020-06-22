////////////////////////////////////////////////////////////////////////////////
//
// opencl_peridynamics.cl (each work item does one bond, but each work group does one node, could be a memory issue with too many bonds)
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
// MAX_HORIZON_LENGTH, PD_R, PD_DX, PD_DT, PD_NODE_NO, PD_DPN_NODE_NO will be defined on JIT compiler's command line



// Update displacements
__kernel void
	UpdateDisplacement(
    	__global double const *Udn,
    	__global double *Un,
		__global int const *BCTypes,
		__global double const *BCValues,
		double DISPLACEMENT_LOAD_SCALE
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{
		Un[i] = (BCTypes[i] == 2 ? (Un[i] + PD_DT * Udn[i]) : (Un[i] + DISPLACEMENT_LOAD_SCALE * BCValues[i]));
	}
}

// Time Integration step
__kernel void
	TimeIntegration(
    __global double const * Un,
    __global double * Udn,
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

        const double _E = Stiffnesses[global_id];
        const double _A = Vols[node_id_j];
        const double _L = xi;

        const double _EAL = _E * _A / _L;

        // Copy bond forces into local memory
        local_cache_x[local_id] = _EAL * cx * y_xi;
        local_cache_y[local_id] = _EAL * cy * y_xi;
        local_cache_z[local_id] = _EAL * cz * y_xi;

        // Check for state of bonds here, and break it if necessary
        const double PD_S0 = FailStretches[global_id];
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
        Udn[DPN * node_no + 0] = (FCTypes[DPN * node_no + 0] == 2 ? local_cache_x[0] : (local_cache_x[0] + FORCE_LOAD_SCALE * FCValues[DPN * node_no + 0]));
        Udn[DPN * node_no + 1] = (FCTypes[DPN * node_no + 1] == 2 ? local_cache_y[0] : (local_cache_y[0] + FORCE_LOAD_SCALE * FCValues[DPN * node_no + 1]));
        Udn[DPN * node_no + 2] = (FCTypes[DPN * node_no + 2] == 2 ? local_cache_z[0] : (local_cache_z[0] + FORCE_LOAD_SCALE * FCValues[DPN * node_no + 2]));
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

__kernel void
	CheckBonds(
		__global int *Horizons,
		__global double const *Un,
		__global double const *Nodes,
		__global double const *FailStretches
	)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if ((i < PD_NODE_NO) && (j >= 0) && (j < MAX_HORIZON_LENGTH))
	{
		const int n = Horizons[i * MAX_HORIZON_LENGTH + j];

		if (n != -1)
		{
			const double xi_x = Nodes[DPN * n + 0] - Nodes[DPN * i + 0];  // Optimize later
			const double xi_y = Nodes[DPN * n + 1] - Nodes[DPN * i + 1];
			const double xi_z = Nodes[DPN * n + 2] - Nodes[DPN * i + 2];

			const double xi_eta_x = Un[DPN * n + 0] - Un[DPN * i + 0] + xi_x;
			const double xi_eta_y = Un[DPN * n + 1] - Un[DPN * i + 1] + xi_y;
			const double xi_eta_z = Un[DPN * n + 2] - Un[DPN * i + 2] + xi_z;

			const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
			const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);

			const double PD_S0 = FailStretches[i * MAX_HORIZON_LENGTH + j];

			const double s = (y - xi) / xi;

			// Check for state of the bond

			if (s > PD_S0)
			{
				Horizons[i * MAX_HORIZON_LENGTH + j] = -1;  // Break the bond
			}
		}
	}
}