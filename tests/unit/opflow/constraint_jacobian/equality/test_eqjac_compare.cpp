#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <exago_config.h>
#include <opflow.h>
#include <private/opflowimpl.h>

#if defined(EXAGO_ENABLE_RAJA)
#include <RAJA/RAJA.hpp>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#endif

struct TripletEntry {
  int row, col;
  double val;
};

static void computeReferenceJacobian(OPFLOW opflow, Vec X,
                                     std::vector<TripletEntry> &entries) {
  PetscErrorCode ierr;
  ierr = (*opflow->modelops.computeequalityconstraintjacobian)(opflow, X,
                                                               opflow->Jac_Ge);

  PetscInt nrow, ncol;
  ierr = MatGetSize(opflow->Jac_Ge, &nrow, &ncol);

  for (PetscInt i = 0; i < nrow; i++) {
    PetscInt nvals;
    const PetscInt *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(opflow->Jac_Ge, i, &nvals, &cols, &vals);
    for (PetscInt j = 0; j < nvals; j++) {
      entries.push_back({(int)i, (int)cols[j], vals[j]});
    }
    ierr = MatRestoreRow(opflow->Jac_Ge, i, &nvals, &cols, &vals);
  }
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscBool flg;
  char file_c_str[PETSC_MAX_PATH_LEN];
  std::string file;
  char appname[] = "opflow";
  MPI_Comm comm = MPI_COMM_WORLD;

  char help[] = "Compare PETSc vs GPU equality constraint Jacobian\n";

  ierr = ExaGOInitialize(comm, &argc, &argv, appname, help);
  if (ierr) {
    fprintf(stderr, "Could not initialize ExaGO.\n");
    return ierr;
  }

  ierr = PetscOptionsGetString(NULL, NULL, "-netfile", file_c_str,
                               PETSC_MAX_PATH_LEN, &flg);
  if (!flg)
    file = "../datafiles/case9/case9mod.m";
  else
    file.assign(file_c_str);

  /* ----------------------------------------------------------------
   * Step 1: Set up OPFLOW with PBPOL model (PETSc path) to get
   *         the reference Jacobian and the initial guess X.
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

  std::vector<TripletEntry> ref_entries;
  computeReferenceJacobian(opflow_ref, X_ref, ref_entries);

  /* ----------------------------------------------------------------
   * Step 2: Set up OPFLOW with HIOPSPARSE to exercise the GPU path.
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

  /* Get the initial guess vector and map to sparse-dense ordering */
  Vec X_gpu;
  ierr = OPFLOWGetSolution(opflow_gpu, &X_gpu);
  CHKERRQ(ierr);

  int nx = opflow_gpu->nx;
  int nnz_eq = opflow_gpu->nnz_eqjacsp;

  printf("\n");
  printf("============================================================\n");
  printf("  Equality Constraint Jacobian: PETSc vs GPU Comparison\n");
  printf("  Network: %s\n", file.c_str());
  printf("  nx = %d, nconeq = %d, nnz_eqjac(GPU) = %d, nnz_eqjac(PETSc) = %d\n",
         nx, opflow_gpu->nconeq, nnz_eq, (int)ref_entries.size());
  printf("============================================================\n\n");

#if defined(EXAGO_ENABLE_RAJA)
  auto &resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator h_allocator = resmgr.getAllocator("HOST");

  double *x_host;
  ierr = VecGetArray(X_gpu, &x_host);
  CHKERRQ(ierr);

  int *iRow, *jCol;
  double *values;
  iRow = static_cast<int *>(h_allocator.allocate(nnz_eq * sizeof(int)));
  jCol = static_cast<int *>(h_allocator.allocate(nnz_eq * sizeof(int)));
  values = static_cast<double *>(h_allocator.allocate(nnz_eq * sizeof(double)));

#ifdef EXAGO_ENABLE_GPU
  umpire::Allocator d_allocator = resmgr.getAllocator("DEVICE");

  double *x_dev =
      static_cast<double *>(d_allocator.allocate(nx * sizeof(double)));
  int *iRow_dev =
      static_cast<int *>(d_allocator.allocate(nnz_eq * sizeof(int)));
  int *jCol_dev =
      static_cast<int *>(d_allocator.allocate(nnz_eq * sizeof(int)));
  double *values_dev =
      static_cast<double *>(d_allocator.allocate(nnz_eq * sizeof(double)));

  umpire::util::AllocationRecord record_x{
      x_host, sizeof(double) * nx, h_allocator.getAllocationStrategy()};
  resmgr.registerAllocation(x_host, record_x);
  resmgr.copy(x_dev, x_host);
#else
  double *x_dev = x_host;
  int *iRow_dev = iRow;
  int *jCol_dev = jCol;
  double *values_dev = values;
#endif

  /* Call the sparse eq jac function: first for sparsity, then for values */
  ierr = (*opflow_gpu->modelops.computesparseequalityconstraintjacobianhiop)(
      opflow_gpu, x_dev, iRow_dev, jCol_dev, NULL);
  CHKERRQ(ierr);

  ierr = (*opflow_gpu->modelops.computesparseequalityconstraintjacobianhiop)(
      opflow_gpu, x_dev, NULL, NULL, values_dev);
  CHKERRQ(ierr);

#ifdef EXAGO_ENABLE_GPU
  resmgr.copy(iRow, iRow_dev);
  resmgr.copy(jCol, jCol_dev);
  resmgr.copy(values, values_dev);
#endif

  /* Build a map from (row, col) -> value for the GPU result */
  struct PairHash {
    size_t operator()(const std::pair<int, int> &p) const {
      return std::hash<long long>()(((long long)p.first << 32) | p.second);
    }
  };
  std::unordered_map<std::pair<int, int>, double, PairHash> gpu_map;
  for (int i = 0; i < nnz_eq; i++) {
    auto key = std::make_pair(iRow[i], jCol[i]);
    gpu_map[key] += values[i];
  }

  /* ----------------------------------------------------------------
   * Step 3: Compare and print results
   * ---------------------------------------------------------------- */
  int n_match = 0, n_mismatch = 0, n_missing_gpu = 0, n_extra_gpu = 0;
  double max_abs_err = 0.0, max_rel_err = 0.0;
  int worst_row = -1, worst_col = -1;
  double worst_ref = 0, worst_gpu = 0;
  const double tol = 1e-6;

  printf("  %-8s %-8s %16s %16s %12s  %s\n", "Row", "Col", "PETSc (ref)",
         "GPU", "AbsErr", "Status");
  printf("  %-8s %-8s %16s %16s %12s  %s\n", "---", "---", "-----------",
         "---", "------", "------");

  for (const auto &e : ref_entries) {
    auto key = std::make_pair(e.row, e.col);
    auto it = gpu_map.find(key);
    double gpu_val = 0.0;
    bool found = (it != gpu_map.end());

    if (found) {
      gpu_val = it->second;
      gpu_map.erase(it);
    }

    double abs_err = fabs(gpu_val - e.val);
    double rel_err =
        (fabs(e.val) > 1e-12) ? abs_err / fabs(e.val) : abs_err;
    const char *status;

    if (!found) {
      status = "MISSING";
      n_missing_gpu++;
    } else if (abs_err < tol) {
      status = "OK";
      n_match++;
    } else {
      status = "MISMATCH";
      n_mismatch++;
    }

    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
      max_rel_err = rel_err;
      worst_row = e.row;
      worst_col = e.col;
      worst_ref = e.val;
      worst_gpu = gpu_val;
    }

    if (abs_err >= tol || !found) {
      printf("  %-8d %-8d %16.8e %16.8e %12.2e  %s\n", e.row, e.col, e.val,
             gpu_val, abs_err, status);
    }
  }

  n_extra_gpu = (int)gpu_map.size();
  if (n_extra_gpu > 0) {
    printf("\n  Extra entries in GPU (not in PETSc reference):\n");
    for (const auto &kv : gpu_map) {
      printf("  %-8d %-8d %16s %16.8e %12s  EXTRA\n", kv.first.first,
             kv.first.second, "n/a", kv.second, "n/a");
    }
  }

  printf("\n");
  printf("============================================================\n");
  printf("  SUMMARY\n");
  printf("============================================================\n");
  printf("  PETSc nnz:         %d\n", (int)ref_entries.size());
  printf("  GPU nnz:           %d\n", nnz_eq);
  printf("  Matching:          %d\n", n_match);
  printf("  Mismatched:        %d\n", n_mismatch);
  printf("  Missing in GPU:    %d\n", n_missing_gpu);
  printf("  Extra in GPU:      %d\n", n_extra_gpu);
  printf("  Max absolute err:  %.2e  at (%d, %d)  ref=%.8e  gpu=%.8e\n",
         max_abs_err, worst_row, worst_col, worst_ref, worst_gpu);
  printf("  Max relative err:  %.2e\n", max_rel_err);
  printf("  Tolerance:         %.2e\n", tol);
  printf("  RESULT:            %s\n",
         (n_mismatch == 0 && n_missing_gpu == 0 && n_extra_gpu == 0)
             ? "PASS"
             : "FAIL");
  printf("============================================================\n\n");

  int result =
      (n_mismatch == 0 && n_missing_gpu == 0 && n_extra_gpu == 0) ? 0 : 1;

  h_allocator.deallocate(iRow);
  h_allocator.deallocate(jCol);
  h_allocator.deallocate(values);
#ifdef EXAGO_ENABLE_GPU
  d_allocator.deallocate(x_dev);
  d_allocator.deallocate(iRow_dev);
  d_allocator.deallocate(jCol_dev);
  d_allocator.deallocate(values_dev);
#endif

  ierr = VecRestoreArray(X_gpu, &x_host);
  CHKERRQ(ierr);
#endif // EXAGO_ENABLE_RAJA

  ierr = OPFLOWDestroy(&opflow_ref);
  CHKERRQ(ierr);
  ierr = OPFLOWDestroy(&opflow_gpu);
  CHKERRQ(ierr);

  ExaGOFinalize();

#if defined(EXAGO_ENABLE_RAJA)
  return result;
#else
  return 0;
#endif
}
