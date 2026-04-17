#include <iostream>
#include <cstdio>
#include <string>

#include <private/opflowimpl.h>
#include <exago_config.h>
#include <utils.h>

#include "opflow_tests.h"
#include "test_acopf_utils.h"

PetscErrorCode ConstructReferenceJacobian(Mat* J, int num_copies)
{
  PetscFunctionBeginUser;
  std::vector<int> i_base = {
      0, 0, 0, 0,
      1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4,
      5, 5, 5, 5, 5,
      6, 6, 6, 6, 6, 6,
      7, 7, 7, 7, 7, 7,
      8, 8, 8, 8,
      9, 9, 9, 9
  };
  std::vector<int> j_base = {
      0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3, 4, 5, 8, 9,
      0, 1, 2, 3, 4, 5, 8, 9,
      2, 3, 4, 5, 6,
      2, 3, 4, 5, 7,
      2, 3, 8, 9, 10, 11,
      2, 3, 8, 9, 10, 11,
      8, 9, 10, 11,
      8, 9, 10, 11
  };
  std::vector<double> v_base = {
      0.8, 0.8, -0.8, -0.8,
      -1.6, -2.0, 1.6, -0.4,
      -0.8, -0.8, 0.8, 2.8, 0.8, -0.2, -0.8, -0.8,
      1.6, -0.4, -3.6, -3.8, 0.4, 0.4, 1.6, -0.4,
      -0.8, 0.2, 0.8, 1.8, -1.0,
      -0.4, -0.4, 0.4, -2.0, -1.0,
      -0.8, -0.8, 1.6, 1.6, -0.8, -0.8,
      1.6, -0.4, -3.2, -4.0, 1.6, -0.4,
      -0.8, -0.8, 0.8, 0.8,
      1.6, -0.4, -1.6, -2.0
  };

  int nrows_base = 10;
  int ncols_base = 12;
  int nrows = (nrows_base - 2) * num_copies + 2;
  int ncols = (ncols_base - 2) * num_copies + 2;

  std::vector<int> i_coo = i_base;
  std::vector<int> j_coo = j_base;
  std::vector<double> v_coo = v_base;

  for (int n = 1; n < num_copies; ++n) {
    auto row_start = n * (nrows_base - 2);
    auto col_start = n * (ncols_base - 2);
    for (int c = 0; c < v_base.size(); ++c) {
      i_coo.push_back(row_start + i_base[c]);
      j_coo.push_back(col_start + j_base[c]);
      v_coo.push_back(v_base[c]);
    }
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD, J));
  PetscCall(MatSetSizes(*J, nrows, ncols, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(*J, MATSEQAIJ));
  PetscCall(MatSetPreallocationCOO(*J, v_coo.size(), i_coo.data(), j_coo.data()));
  PetscCall(MatSetValuesCOO(*J, v_coo.data(), ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Unit test driver for the "compute equality constraint jacobian"
 * @see opflow/OpflowTests.hpp for kernel tested by this driver
 */
int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscBool flg;
  Vec X;
  int fail = 0;
  double obj_value;
  char file_c_str[PETSC_MAX_PATH_LEN];
  char validation_c_str[PETSC_MAX_PATH_LEN];
  std::string file;
  char appname[] = "opflow";
  MPI_Comm comm = MPI_COMM_WORLD;
  int num_copies = 0;

  char help[] = "Unit tests for comparing GPU- and CPU-computed equality constraint jacobians\n";

  /** Use `ExaGOLogSetLoggingFileName("opflow-logfile");` to log the output. */
  ierr = ExaGOInitialize(comm, &argc, &argv, appname, help);
  if (ierr) {
    fprintf(stderr, "Could not initialize ExaGO application %s.\n", appname);
    return ierr;
  }

  /* Get network data file from command line */
  ierr = PetscOptionsGetString(NULL, NULL, "-netfile", file_c_str,
                               PETSC_MAX_PATH_LEN, &flg);
  CHKERRQ(ierr);

  /* Get num_copies from command line */
  ierr = PetscOptionsGetInt(NULL, NULL, "-num_copies", &num_copies, &flg);
  CHKERRQ(ierr);

  Mat J_eq_ref;
  ConstructReferenceJacobian(&J_ref, num_copies);

  if (!flg) {
    file = "CECJ_unittest1.m";
  } else {
    file.assign(file_c_str);
  }

  OPFLOW opflowtest;
  exago::tests::TestOpflow test;

  /* Set up test opflow */
  ierr = OPFLOWCreate(PETSC_COMM_WORLD, &opflowtest);
  CHKERRQ(ierr);
  ierr = OPFLOWReadMatPowerData(opflowtest, file.c_str());
  CHKERRQ(ierr);
  ierr = OPFLOWSetInitializationType(opflowtest, OPFLOWINIT_FROMFILE);
  CHKERRQ(ierr);
  ierr = OPFLOWSetUp(opflowtest);
  CHKERRQ(ierr);
  ierr = OPFLOWGetSolution(opflowtest, &X);
  CHKERRQ(ierr);

  // If we are using HIOP, need to convert X
  // The string lengths must be 65
  std::string modelname;
  std::string solvername;
  ierr = OPFLOWGetModel(opflowtest, &modelname);
  ierr = OPFLOWGetSolver(opflowtest, &solvername);

  if (solvername == "HIOP") {
#if defined(EXAGO_ENABLE_HIOP)
    double *x_ref;
    ierr = VecGetArray(X, &x_ref);

    int nx, nconeq, nconineq;
    ierr = OPFLOWGetSizes(opflowtest, &nx, &nconeq, &nconineq);
    CHKERRQ(ierr);

    // If we are running using the CPU model, nothing needs to be done
    if (modelname == "POWER_BALANCE_HIOP") {
      fail += test.computeConstraintJacobian(opflowtest, x_ref, J_eq_ref);
    } else // Using model PBPOLRAJAHIOP
    {
#if defined(EXAGO_ENABLE_RAJA)
      // Get resource manager instance
      auto &resmgr = umpire::ResourceManager::getInstance();

      // Get Allocator
      umpire::Allocator h_allocator = resmgr.getAllocator("HOST");

      // Register array xref with umpire
      umpire::util::AllocationRecord record_x{
          x_ref, sizeof(double) * nx, h_allocator.getAllocationStrategy()};
      resmgr.registerAllocation(x_ref, record_x);
      // Allocate and copy xref to device
      double *x_ref_dev;

#ifdef EXAGO_ENABLE_GPU

      ierr = OPFLOWSetHIOPComputeMode(opflowtest, "GPU");
      CHKERRQ(ierr);

      umpire::Allocator d_allocator = resmgr.getAllocator("DEVICE");
      x_ref_dev =
          static_cast<double *>(d_allocator.allocate(nx * sizeof(double)));
#else
      ierr = OPFLOWSetHIOPComputeMode(opflowtest, "CPU");
      CHKERRQ(ierr);
      x_ref_dev = x_ref;
#endif
      resmgr.copy(x_ref_dev, x_ref);

      fail += test.computeObjective(opflowtest, x_ref_dev, obj_value);

#ifdef EXAGO_ENABLE_GPU
      d_allocator.deallocate(x_ref_dev);
#endif
#endif
    }

    ierr = VecRestoreArray(X, &x_ref);
    CHKERRQ(ierr);

    ierr = PetscFree(x_ref);
    CHKERRQ(ierr);
#endif // End #ifdefined(EXAGO_ENABLE_HIOP)
  } else {
    fail += test.computeObjective(opflowtest, X, obj_value);
  }
  ierr = OPFLOWDestroy(&opflowtest);
  CHKERRQ(ierr);

  ExaGOFinalize();
  return fail;
}
