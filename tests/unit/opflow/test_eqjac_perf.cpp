#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

#include <exago_config.h>
#include <opflow.h>
#include <private/opflowimpl.h>

#if defined(EXAGO_ENABLE_RAJA)
#include <RAJA/RAJA.hpp>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#endif

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double benchmarkPETSc(OPFLOW opflow, Vec X, int niters) {
  PetscErrorCode ierr;
  PetscScalar *x_arr;

  ierr = VecGetArray(X, &x_arr);
  auto t0 = Clock::now();
  for (int iter = 0; iter < niters; iter++) {
    ierr = (*opflow->modelops.computeequalityconstraintjacobian)(
        opflow, X, opflow->Jac_Ge);
  }
  auto t1 = Clock::now();
  ierr = VecRestoreArray(X, &x_arr);

  return Ms(t1 - t0).count() / niters;
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscBool flg;
  char file_c_str[PETSC_MAX_PATH_LEN];
  std::string file;
  char appname[] = "opflow";
  MPI_Comm comm = MPI_COMM_WORLD;
  int niters = 100;

  char help[] = "Benchmark PETSc vs GPU equality constraint Jacobian\n";

  ierr = ExaGOInitialize(comm, &argc, &argv, appname, help);
  if (ierr)
    return ierr;

  ierr = PetscOptionsGetString(NULL, NULL, "-netfile", file_c_str,
                               PETSC_MAX_PATH_LEN, &flg);
  if (!flg)
    file = "../datafiles/case9/case9mod.m";
  else
    file.assign(file_c_str);

  ierr = PetscOptionsGetInt(NULL, NULL, "-niters", &niters, &flg);

  /* ----------------------------------------------------------------
   * PETSc path: IPOPT + PBPOL
   * ---------------------------------------------------------------- */
  OPFLOW opflow_ref;
  ierr = OPFLOWCreate(PETSC_COMM_WORLD, &opflow_ref);
  CHKERRQ(ierr);
  ierr = OPFLOWReadMatPowerData(opflow_ref, file.c_str());
  CHKERRQ(ierr);
  ierr = OPFLOWSetModel(opflow_ref, OPFLOWMODEL_PBPOL);
  CHKERRQ(ierr);
  ierr = OPFLOWSetSolver(opflow_ref, OPFLOWSOLVER_IPOPT);
  CHKERRQ(ierr);
  ierr = OPFLOWSetInitializationType(opflow_ref, OPFLOWINIT_FROMFILE);
  CHKERRQ(ierr);
  ierr = OPFLOWSetUp(opflow_ref);
  CHKERRQ(ierr);

  Vec X_ref;
  ierr = OPFLOWGetSolution(opflow_ref, &X_ref);
  CHKERRQ(ierr);

  /* Warmup */
  benchmarkPETSc(opflow_ref, X_ref, 5);
  double petsc_ms = benchmarkPETSc(opflow_ref, X_ref, niters);

  int petsc_nnz = 0;
  {
    MatInfo info;
    ierr = MatGetInfo(opflow_ref->Jac_Ge, MAT_LOCAL, &info);
    petsc_nnz = (int)info.nz_used;
  }

  /* ----------------------------------------------------------------
   * GPU path: HIOPSPARSEGPU + PBPOLRAJAHIOPSPARSE
   * ---------------------------------------------------------------- */
  OPFLOW opflow_gpu;
  ierr = OPFLOWCreate(PETSC_COMM_WORLD, &opflow_gpu);
  CHKERRQ(ierr);
  ierr = OPFLOWReadMatPowerData(opflow_gpu, file.c_str());
  CHKERRQ(ierr);
  ierr = OPFLOWSetModel(opflow_gpu, OPFLOWMODEL_PBPOLRAJAHIOPSPARSE);
  CHKERRQ(ierr);
  ierr = OPFLOWSetSolver(opflow_gpu, OPFLOWSOLVER_HIOPSPARSEGPU);
  CHKERRQ(ierr);
  ierr = OPFLOWSetInitializationType(opflow_gpu, OPFLOWINIT_FROMFILE);
  CHKERRQ(ierr);
#ifdef EXAGO_ENABLE_GPU
  ierr = OPFLOWSetHIOPComputeMode(opflow_gpu, "GPU");
  CHKERRQ(ierr);
#endif
  ierr = OPFLOWSetUp(opflow_gpu);
  CHKERRQ(ierr);

  Vec X_gpu;
  ierr = OPFLOWGetSolution(opflow_gpu, &X_gpu);
  CHKERRQ(ierr);

  int nx = opflow_gpu->nx;
  int nnz_eq = opflow_gpu->nnz_eqjacsp;

#if defined(EXAGO_ENABLE_RAJA)
  auto &resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator h_allocator = resmgr.getAllocator("HOST");

  double *x_host;
  ierr = VecGetArray(X_gpu, &x_host);
  CHKERRQ(ierr);

  double *values_dev;
  double *x_dev;

#ifdef EXAGO_ENABLE_GPU
  umpire::Allocator d_allocator = resmgr.getAllocator("DEVICE");
  x_dev = static_cast<double *>(d_allocator.allocate(nx * sizeof(double)));
  values_dev =
      static_cast<double *>(d_allocator.allocate(nnz_eq * sizeof(double)));

  umpire::util::AllocationRecord record_x{x_host, sizeof(double) * nx,
                                          h_allocator.getAllocationStrategy()};
  resmgr.registerAllocation(x_host, record_x);
  resmgr.copy(x_dev, x_host);
#else
  x_dev = x_host;
  values_dev =
      static_cast<double *>(h_allocator.allocate(nnz_eq * sizeof(double)));
#endif

  /* Run sparsity phase once (not timed — this is one-time setup) */
  {
    int *iRow_dev, *jCol_dev;
#ifdef EXAGO_ENABLE_GPU
    iRow_dev = static_cast<int *>(d_allocator.allocate(nnz_eq * sizeof(int)));
    jCol_dev = static_cast<int *>(d_allocator.allocate(nnz_eq * sizeof(int)));
#else
    iRow_dev = static_cast<int *>(h_allocator.allocate(nnz_eq * sizeof(int)));
    jCol_dev = static_cast<int *>(h_allocator.allocate(nnz_eq * sizeof(int)));
#endif
    ierr = (*opflow_gpu->modelops.computesparseequalityconstraintjacobianhiop)(
        opflow_gpu, x_dev, iRow_dev, jCol_dev, NULL);
    CHKERRQ(ierr);
#ifdef EXAGO_ENABLE_GPU
    d_allocator.deallocate(iRow_dev);
    d_allocator.deallocate(jCol_dev);
#else
    h_allocator.deallocate(iRow_dev);
    h_allocator.deallocate(jCol_dev);
#endif
  }

  /* Warmup the values kernel */
  for (int i = 0; i < 5; i++) {
    ierr = (*opflow_gpu->modelops.computesparseequalityconstraintjacobianhiop)(
        opflow_gpu, x_dev, NULL, NULL, values_dev);
    CHKERRQ(ierr);
  }

  // HIP kernels do not synchronize by default
#ifdef EXAGO_ENABLE_HIP
  int status = hipDeviceSynchronize();
#endif

  /* Timed runs */
  auto t0 = Clock::now();
  for (int iter = 0; iter < niters; iter++) {
    ierr = (*opflow_gpu->modelops.computesparseequalityconstraintjacobianhiop)(
        opflow_gpu, x_dev, NULL, NULL, values_dev);
    CHKERRQ(ierr);
  }

  // HIP kernels do not synchronize by default
#ifdef EXAGO_ENABLE_HIP
  status = hipDeviceSynchronize();
#endif

  auto t1 = Clock::now();
  double gpu_ms = Ms(t1 - t0).count() / niters;

#ifdef EXAGO_ENABLE_GPU
  d_allocator.deallocate(x_dev);
  d_allocator.deallocate(values_dev);
#else
  h_allocator.deallocate(values_dev);
#endif

  ierr = VecRestoreArray(X_gpu, &x_host);
  CHKERRQ(ierr);
#else
  double gpu_ms = 0.0;
#endif

  /* ----------------------------------------------------------------
   * Print results
   * ---------------------------------------------------------------- */
  printf("\n");
  printf("================================================================\n");
  printf("  Equality constraint Jacobian — performance comparison\n");
  printf("================================================================\n");
  printf("  Network:          %s\n", file.c_str());
  printf("  Buses:            %d\n", opflow_gpu->ps->nbus);
  printf("  Variables (nx):   %d\n", nx);
  printf("  Eq constraints:   %d\n", opflow_gpu->nconeq);
  printf("  nnz (PETSc):      %d\n", petsc_nnz);
  printf("  nnz (GPU):        %d\n", nnz_eq);
  printf("  Iterations:       %d\n", niters);
  printf("----------------------------------------------------------------\n");
  printf("  %-20s %12s %12s\n", "", "PETSc (CPU)", "RAJA (GPU)");
  printf("  %-20s %12s %12s\n", "", "-----------", "----------");
  printf("  %-20s %10.4f ms %10.4f ms\n", "Avg time/call", petsc_ms, gpu_ms);
  if (gpu_ms > 0.0) {
    double speedup = petsc_ms / gpu_ms;
    printf("  %-20s %10s    %9.2fx\n", "Speedup", "", speedup);
  }
  printf(
      "================================================================\n\n");

  ierr = OPFLOWDestroy(&opflow_ref);
  CHKERRQ(ierr);
  ierr = OPFLOWDestroy(&opflow_gpu);
  CHKERRQ(ierr);

  ExaGOFinalize();
  return 0;
}
