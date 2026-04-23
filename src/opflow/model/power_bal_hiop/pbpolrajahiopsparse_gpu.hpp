#include <exago_config.h>

#if defined(EXAGO_ENABLE_RAJA)
#if defined(EXAGO_ENABLE_HIOP_SPARSE)

#pragma once

#include <private/opflowimpl.h>
#include "pbpolrajahiopsparse.hpp"

/**
 * GPU-only (PETSc-free) computation of inequality constraint Jacobian values.
 *
 * Replaces the PETSc-based path that calls
 * opflow->modelops.computeinequalityconstraintjacobian followed by
 * MatGetRow extraction. Writes directly into device memory using RAJA
 * kernels, without PETSc Mat/Vec operations; no H2D, D2H copies back and forth.
 *
 * @param opflow     The OPFLOW problem context
 * @param x_dev      Device array of variable values
 * @param jacd_dev   Device output array for inequality Jacobian values
 *                   (points to the ineq portion of the sparse Jacobian,
 *                    i.e. MJacS_dev + nnz_eqjacsp)
 */
void ComputeIneqJacValuesGPU_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                                 const double *x_dev,
                                                 double *jacd_dev);

/**
 * GPU-only (PETSc-free) computation of equality constraint Jacobian values.
 */
void ComputeEqJacValuesGPU_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                                const double *x_dev,
                                                double *jace_dev);

#endif // EXAGO_ENABLE_HIOP_SPARSE
#endif // EXAGO_ENABLE_RAJA
