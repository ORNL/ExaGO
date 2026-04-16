#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#include <exago_config.h>
#include <private/opflowimpl.h>
#include <utils.h>

#if defined(EXAGO_ENABLE_RAJA)
#include <RAJA/RAJA.hpp>
#include <private/raja_exec_config.h>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#endif

#ifdef EXAGO_ENABLE_GPU
#include <hip/hip_runtime.h>
#endif

#include "model/power_bal_hiop/paramsrajahiop.h"
#include "model/power_bal_hiop/pbpolrajahiopsparse_gpu.hpp"

static const double TOL = 1e-8;

static int compare_arrays(const double *ref, const double *gpu, int n,
                          const char *label) {
  int fail = 0;
  for (int i = 0; i < n; i++) {
    if (std::abs(ref[i] - gpu[i]) / (1.0 + std::abs(ref[i])) > TOL) {
      std::cout << "  MISMATCH " << label << "[" << i << "]: PETSc=" << ref[i]
                << "  GPU=" << gpu[i] << "  diff=" << std::abs(ref[i] - gpu[i])
                << std::endl;
      fail++;
    }
  }
  return fail;
}

/**
 * Test 8: Validate GPU inequality constraint Jacobian values against PETSc.
 *
 * First solves with IPOPT/PBPOL to obtain a realistic solution, then
 * sets up the PBPOLRAJAHIOPSPARSE model and evaluates the inequality
 * Jacobian at the converged solution using both the PETSc reference
 * path and the GPU RAJA kernel. Compares element-by-element.
 */
int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscBool flg;
  char file_c_str[PETSC_MAX_PATH_LEN];
  std::string file;
  char appname[] = "opflow";
  MPI_Comm comm = MPI_COMM_WORLD;
  char help[] = "Test 8: GPU ineq Jacobian validation\n";
  int fail = 0;

  ierr = ExaGOInitialize(comm, &argc, &argv, appname, help);
  if (ierr) {
    fprintf(stderr, "Could not initialize ExaGO.\n");
    return ierr;
  }

  ierr = PetscOptionsGetString(NULL, NULL, "-netfile", file_c_str,
                               PETSC_MAX_PATH_LEN, &flg);
  CHKERRQ(ierr);
  if (!flg)
    file = "../datafiles/case9/case9mod.m";
  else
    file.assign(file_c_str);

  std::cout << "=== Test 8: GPU Inequality Jacobian Validation ==="
            << std::endl;
  std::cout << "Network file: " << file << std::endl;

  /* ------------------------------------------------------------------
   * Step 1: Solve with IPOPT/PBPOL to get a realistic solution
   * ------------------------------------------------------------------ */
  OPFLOW opflow_ref;
  Vec Xsol;

  ierr = OPFLOWCreate(PETSC_COMM_WORLD, &opflow_ref);
  CHKERRQ(ierr);
  ierr = OPFLOWReadMatPowerData(opflow_ref, file.c_str());
  CHKERRQ(ierr);
  ierr = OPFLOWSetModel(opflow_ref, OPFLOWMODEL_PBPOL);
  CHKERRQ(ierr);
  ierr = OPFLOWSetSolver(opflow_ref, OPFLOWSOLVER_IPOPT);
  CHKERRQ(ierr);
  ierr = OPFLOWSolve(opflow_ref);
  CHKERRQ(ierr);
  ierr = OPFLOWGetSolution(opflow_ref, &Xsol);
  CHKERRQ(ierr);

  std::cout << "IPOPT solve complete." << std::endl;

  /* Get the solution array (natural ordering) */
  PetscInt nx_ref;
  ierr = VecGetSize(Xsol, &nx_ref);
  CHKERRQ(ierr);
  double *xsol_nat;
  ierr = VecGetArray(Xsol, &xsol_nat);
  CHKERRQ(ierr);

  /* ------------------------------------------------------------------
   * Step 2: Create the PBPOLRAJAHIOPSPARSE model and set up
   * ------------------------------------------------------------------ */
  OPFLOW opflow;
  ierr = OPFLOWCreate(PETSC_COMM_WORLD, &opflow);
  CHKERRQ(ierr);
  ierr = OPFLOWReadMatPowerData(opflow, file.c_str());
  CHKERRQ(ierr);
  ierr = OPFLOWSetModel(opflow, OPFLOWMODEL_PBPOLRAJAHIOPSPARSE);
  CHKERRQ(ierr);
  ierr = OPFLOWSetSolver(opflow, OPFLOWSOLVER_HIOPSPARSEGPU);
  CHKERRQ(ierr);
  ierr = OPFLOWSetUp(opflow);
  CHKERRQ(ierr);

  int nx, nconeq, nconineq;
  ierr = OPFLOWGetSizes(opflow, &nx, &nconeq, &nconineq);
  CHKERRQ(ierr);

  std::cout << "nx=" << nx << " nconeq=" << nconeq << " nconineq=" << nconineq
            << " nnz_ineqjacsp=" << opflow->nnz_ineqjacsp << std::endl;

  if (!nconineq) {
    std::cout << "No inequality constraints -- nothing to test. PASS."
              << std::endl;
    ierr = VecRestoreArray(Xsol, &xsol_nat);
    CHKERRQ(ierr);
    ierr = OPFLOWDestroy(&opflow_ref);
    CHKERRQ(ierr);
    ierr = OPFLOWDestroy(&opflow);
    CHKERRQ(ierr);
    ExaGOFinalize();
    return 0;
  }

  /* ------------------------------------------------------------------
   * Step 3: Copy IPOPT solution into opflow->X (natural ordering)
   * ------------------------------------------------------------------ */
  double *x_nat;
  ierr = VecGetArray(opflow->X, &x_nat);
  CHKERRQ(ierr);
  for (int i = 0; i < nx; i++)
    x_nat[i] = xsol_nat[i];
  ierr = VecRestoreArray(opflow->X, &x_nat);
  CHKERRQ(ierr);

  ierr = VecRestoreArray(Xsol, &xsol_nat);
  CHKERRQ(ierr);

  std::cout << "Evaluating Jacobian at IPOPT-converged solution." << std::endl;

  /* ------------------------------------------------------------------
   * Step 4: Compute reference inequality Jacobian via PETSc.
   * The first call establishes the sparsity pattern, the second
   * computes the actual values at the converged solution.
   * ------------------------------------------------------------------ */
  ierr = (*opflow->modelops.computeinequalityconstraintjacobian)(
      opflow, opflow->X, opflow->Jac_Gi);
  CHKERRQ(ierr);
  ierr = MatSetOption(opflow->Jac_Gi, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
  CHKERRQ(ierr);
  ierr = (*opflow->modelops.computeinequalityconstraintjacobian)(
      opflow, opflow->X, opflow->Jac_Gi);
  CHKERRQ(ierr);

  int nnz = opflow->nnz_ineqjacsp;

  auto &resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator h_allocator = resmgr.getAllocator("HOST");

  double *ref_vals =
      static_cast<double *>(h_allocator.allocate(nnz * sizeof(double)));

  PetscInt nrow, ncol;
  ierr = MatGetSize(opflow->Jac_Gi, &nrow, &ncol);
  CHKERRQ(ierr);

  double *vptr = ref_vals;
  for (int i = 0; i < nrow; i++) {
    PetscInt nvals;
    const PetscInt *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(opflow->Jac_Gi, i, &nvals, &cols, &vals);
    CHKERRQ(ierr);
    for (int j = 0; j < nvals; j++)
      vptr[j] = vals[j];
    vptr += nvals;
    ierr = MatRestoreRow(opflow->Jac_Gi, i, &nvals, &cols, &vals);
    CHKERRQ(ierr);
  }

  int ref_count = (int)(vptr - ref_vals);
  std::cout << "PETSc extracted " << ref_count << " ineq Jacobian values"
            << " (expected " << nnz << ")" << std::endl;

  if (ref_count != nnz) {
    std::cout << "FAIL: NNZ mismatch! PETSc=" << ref_count
              << " analytical=" << nnz << std::endl;
    fail++;
  }

  /* ------------------------------------------------------------------
   * Step 5: Compute GPU inequality Jacobian at the same solution
   * ------------------------------------------------------------------ */
  double *x_host;
  ierr = VecGetArray(opflow->X, &x_host);
  CHKERRQ(ierr);

  double *x_sd =
      static_cast<double *>(h_allocator.allocate(nx * sizeof(double)));
  for (int i = 0; i < nx; i++)
    x_sd[opflow->idxn2sd_map[i]] = x_host[i];

  ierr = VecRestoreArray(opflow->X, &x_host);
  CHKERRQ(ierr);

  double *gpu_vals;
  double *x_dev, *gpu_vals_dev;

#ifdef EXAGO_ENABLE_GPU
  umpire::Allocator d_allocator = resmgr.getAllocator("DEVICE");
  x_dev = static_cast<double *>(d_allocator.allocate(nx * sizeof(double)));
  gpu_vals_dev =
      static_cast<double *>(d_allocator.allocate(nnz * sizeof(double)));
  resmgr.memset(gpu_vals_dev, 0, nnz * sizeof(double));
#else
  x_dev = x_sd;
  gpu_vals_dev =
      static_cast<double *>(h_allocator.allocate(nnz * sizeof(double)));
  memset(gpu_vals_dev, 0, nnz * sizeof(double));
#endif

  umpire::util::AllocationRecord rec_x{x_sd, sizeof(double) * nx,
                                       h_allocator.getAllocationStrategy()};
  resmgr.registerAllocation(x_sd, rec_x);
#ifdef EXAGO_ENABLE_GPU
  resmgr.copy(x_dev, x_sd);
#endif

  std::cout << "Running RAJA GPU inequality Jacobian kernel..." << std::endl;
  ComputeIneqJacValuesGPU_PBPOLRAJAHIOPSPARSE(opflow, x_dev, gpu_vals_dev);

  gpu_vals = static_cast<double *>(h_allocator.allocate(nnz * sizeof(double)));
#ifdef EXAGO_ENABLE_GPU
  resmgr.copy(gpu_vals, gpu_vals_dev);
#else
  memcpy(gpu_vals, gpu_vals_dev, nnz * sizeof(double));
#endif

  /* ------------------------------------------------------------------
   * Step 6: Compare
   * ------------------------------------------------------------------ */
  std::cout << "Comparing " << nnz << " inequality Jacobian values..."
            << std::endl;
  int cmp_fail = compare_arrays(ref_vals, gpu_vals, nnz, "ineqjac");
  fail += cmp_fail;

  if (cmp_fail == 0)
    std::cout << "PASS: All " << nnz
              << " inequality Jacobian values match within tol=" << TOL
              << std::endl;
  else
    std::cout << "FAIL: " << cmp_fail << " of " << nnz << " values differ"
              << std::endl;

  /* ------------------------------------------------------------------
   * Step 7: Performance comparison (enabled with -benchmark flag)
   * ------------------------------------------------------------------ */
  PetscBool run_benchmark = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-benchmark", NULL, &run_benchmark);
  CHKERRQ(ierr);

  if (run_benchmark) {
    int niters = 1000;
    PetscInt bench_nrow, bench_ncol;
    ierr = MatGetSize(opflow->Jac_Gi, &bench_nrow, &bench_ncol);
    CHKERRQ(ierr);

    double *bench_vals =
        static_cast<double *>(h_allocator.allocate(nnz * sizeof(double)));

    std::cout << "\n=== Performance Benchmark (" << niters
              << " iterations) ===" << std::endl;

    /* --- PETSc path: compute + MatGetRow extraction + copy to device --- */
    {
      double *bench_dev;
      size_t nnz_bytes = nnz * sizeof(double);
#ifdef EXAGO_ENABLE_GPU
      bench_dev = static_cast<double *>(d_allocator.allocate(nnz_bytes));
#else
      bench_dev = static_cast<double *>(h_allocator.allocate(nnz_bytes));
#endif

      auto t0 = std::chrono::high_resolution_clock::now();
      for (int iter = 0; iter < niters; iter++) {
        ierr = (*opflow->modelops.computeinequalityconstraintjacobian)(
            opflow, opflow->X, opflow->Jac_Gi);

        double *vp = bench_vals;
        for (int i = 0; i < bench_nrow; i++) {
          PetscInt nv;
          const PetscInt *c;
          const PetscScalar *v;
          MatGetRow(opflow->Jac_Gi, i, &nv, &c, &v);
          for (int j = 0; j < nv; j++)
            vp[j] = v[j];
          vp += nv;
          MatRestoreRow(opflow->Jac_Gi, i, &nv, &c, &v);
        }
#ifdef EXAGO_ENABLE_GPU
        (void)hipMemcpy(bench_dev, bench_vals, nnz_bytes,
                        hipMemcpyHostToDevice);
#else
        memcpy(bench_dev, bench_vals, nnz_bytes);
#endif
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double petsc_us =
          std::chrono::duration<double, std::micro>(t1 - t0).count() / niters;
      std::cout << "  PETSc path (compute + MatGetRow + copy): " << petsc_us
                << " us/iter" << std::endl;

#ifdef EXAGO_ENABLE_GPU
      d_allocator.deallocate(bench_dev);
#else
      h_allocator.deallocate(bench_dev);
#endif
    }

    /* --- GPU path: RAJA kernels, no copies --- */
    {
      double *bench_dev;
#ifdef EXAGO_ENABLE_GPU
      bench_dev =
          static_cast<double *>(d_allocator.allocate(nnz * sizeof(double)));
#else
      bench_dev =
          static_cast<double *>(h_allocator.allocate(nnz * sizeof(double)));
#endif

#ifdef EXAGO_ENABLE_GPU
      (void)hipDeviceSynchronize();
#endif
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int iter = 0; iter < niters; iter++) {
        ComputeIneqJacValuesGPU_PBPOLRAJAHIOPSPARSE(opflow, x_dev, bench_dev);
      }
#ifdef EXAGO_ENABLE_GPU
      (void)hipDeviceSynchronize();
#endif
      auto t1 = std::chrono::high_resolution_clock::now();
      double gpu_us =
          std::chrono::duration<double, std::micro>(t1 - t0).count() / niters;
      std::cout << "  GPU path (RAJA kernels, no copies):      " << gpu_us
                << " us/iter" << std::endl;

#ifdef EXAGO_ENABLE_GPU
      d_allocator.deallocate(bench_dev);
#else
      h_allocator.deallocate(bench_dev);
#endif
    }

    h_allocator.deallocate(bench_vals);
    std::cout << "=== End Benchmark ===" << std::endl;
  }

  /* ------------------------------------------------------------------
   * Cleanup
   * ------------------------------------------------------------------ */
  h_allocator.deallocate(ref_vals);
  h_allocator.deallocate(gpu_vals);
  h_allocator.deallocate(x_sd);
#ifdef EXAGO_ENABLE_GPU
  d_allocator.deallocate(x_dev);
  d_allocator.deallocate(gpu_vals_dev);
#else
  h_allocator.deallocate(gpu_vals_dev);
#endif

  ierr = OPFLOWDestroy(&opflow);
  CHKERRQ(ierr);
  ierr = OPFLOWDestroy(&opflow_ref);
  CHKERRQ(ierr);
  ExaGOFinalize();

  return fail;
}
