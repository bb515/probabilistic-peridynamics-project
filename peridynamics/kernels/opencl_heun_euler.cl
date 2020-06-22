////////////////////////////////////////////////////////////////////////////////
//
// opencl_peridynamics.cl
//
// OpenCL Peridynamics kernels
//
// Based on code from Copyright (c) Farshid Mossaiby, 2016, 2017. Adapted for python.
//
////////////////////////////////////////////////////////////////////////////////

// Includes, project

#include "opencl_enable_fp64.cl"

// Macros

#define DPN 3
// MAX_HORIZON_LENGTH, PD_DT, PD_E, PD_S0, PD_NODE_NO, PD_DPN_NODE_NO will be defined on JIT compiler's command line

// A horizon by horizon approach is chosen to proceed with the solution, in which
// no assembly of the system of equations is required.



// Calculate force using un, force BC applied at end here
__kernel void
	CalcBondForce(
        __global double *Udn,
        __global double const *Un,
        __global double const *Vols,
		__global int const *Horizons,
		__global double const *Nodes,
		__global double const *Stiffnesses,
		__global int const *FCTypes,
		__global double const *FCValues,
		double FORCE_LOAD_SCALE
	)
{
	const int i = get_global_id(0);

	double f0 = 0.00;
	double f1 = 0.00;
	double f2 = 0.00;

	if (i < PD_NODE_NO)
	{
		for (int j = 0; j < MAX_HORIZON_LENGTH; j++)
		{
			const int n = Horizons[MAX_HORIZON_LENGTH * i + j];

			if (n != -1)
			{
				const double xi_x = Nodes[DPN * n + 0] - Nodes[DPN * i + 0];  // Optimize later, doesn't need to be done every time
				const double xi_y = Nodes[DPN * n + 1] - Nodes[DPN * i + 1];
				const double xi_z = Nodes[DPN * n + 2] - Nodes[DPN * i + 2];


				const double xi_eta_x = Un[DPN * n + 0] - Un[DPN * i + 0] + xi_x;
				const double xi_eta_y = Un[DPN * n + 1] - Un[DPN * i + 1] + xi_y;
				const double xi_eta_z = Un[DPN * n + 2] - Un[DPN * i + 2] + xi_z;

				const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
				const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
                const double y_xi = (y - xi);

				const double cx = xi_eta_x / y;
				const double cy = xi_eta_y / y;
				const double cz = xi_eta_z / y;

				const double _E = Stiffnesses[MAX_HORIZON_LENGTH * i + j];
                const double _A = Vols[i];
				const double _L = xi;

				const double _EAL = _E * _A / _L;

                f0 += _EAL * cx * y_xi;
                f1 += _EAL * cy * y_xi;
                f2 += _EAL * cz * y_xi;
			}
		}

		// Add body forces

		f0 = (FCTypes[DPN*i + 0] == 2 ? f0 : (f0 + FORCE_LOAD_SCALE * FCValues[DPN * i + 0]));
		f1 = (FCTypes[DPN*i + 1] == 2 ? f1 : (f1 + FORCE_LOAD_SCALE * FCValues[DPN * i + 1]));
		f2 = (FCTypes[DPN*i + 2] == 2 ? f2 : (f2 + FORCE_LOAD_SCALE * FCValues[DPN * i + 2]));
		
		Udn[DPN * i + 0] = f0;
		Udn[DPN * i + 1] = f1;
		Udn[DPN * i + 2] = f2;
	}
}

// Update displacements using k1 through k4
__kernel void
	UpdateDisplacement(
        __global int const *ICTypes,
		__global double const *ICValues,
		__global double const *Un2_1,
        __global double *Un2,
		double PD_DT
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{
        Un2[i] = ICTypes[i] == 2 ? Un2_1[i]  : Un2[i] + ICValues[i];
	}
}

// Partial update of displacement
__kernel void
	PartialUpdateDisplacement(
        __global double const *Udn,
        __global double const *Un2,
		__global double *Un1,
		double PD_DT
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{
		Un1[i] = Un2[i] + PD_DT * Udn[i];
	}
}

// Check error size
__kernel void
	CheckError(
		__global double const *k1dn,
        __global double const *k2dn,
        __global double const *Un1,
		__global double const *Un2,
		__global double *Un2_1,
        __global double *En,
		double PD_DT
	)
{
	const int i = get_global_id(0);
	double U2_1;
	double U_1;
	if (i < PD_DPN_NODE_NO)
	{
		// 2nd order accurate displacement
		U2_1 = Un2[i] + (1.00 / 2.00) * PD_DT * (k1dn[i] + k2dn[i]);
		// error
		En[i] = U2_1 - Un1[i];
		// store 2nd order displacements
		Un2_1[i] = U2_1;
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

__kernel void
	CalculateDamage(
		__global double *Phi,
		__global int const *Horizons,
		__global int const *HorizonLengths
	)
{
	const int i = get_global_id(0);

	if (i < PD_NODE_NO)
	{
		int active_bonds = 0;

		for (int j = 0; j < MAX_HORIZON_LENGTH; j++)
		{
			if (Horizons[MAX_HORIZON_LENGTH * i + j] != -1)
			{
				active_bonds++;
			}
		}

		Phi[i] = 1.00 - (double) active_bonds / (double) (HorizonLengths[i]);
	}
}