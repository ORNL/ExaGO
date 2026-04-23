#include <exago_config.h>

#if defined(EXAGO_ENABLE_RAJA)
#if defined(EXAGO_ENABLE_HIOP_SPARSE)

#pragma once

#include <private/opflowimpl.h>
#include "pbpolrajahiopsparse.hpp"

/**
 * GPU-only (PETSc-free) computation of equality constraint Jacobian values.
 *
 * Replaces the PETSc-based path that calls
 * opflow->modelops.computeequalityconstraintjacobian followed by
 * MatGetRow extraction. Writes directly into device memory using RAJA
 * kernels, without PETSc Mat/Vec operations; no H2D, D2H copies back and forth.
 *
 * @param opflow     The OPFLOW problem context
 * @param x_dev      Device array of variable values
 * @param jace_dev   Device output array for equality Jacobian values
 */
void ComputeEqJacValuesGPU_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                               const double *x_dev,
                                               double *jace_dev);

#endif // EXAGO_ENABLE_HIOP_SPARSE
#endif // EXAGO_ENABLE_RAJA
