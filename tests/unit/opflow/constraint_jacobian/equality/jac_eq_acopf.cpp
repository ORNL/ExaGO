#include <iostream>
#include <cstdio>
#include <numeric>
#include <string>

#include <private/opflowimpl.h>
#include <exago_config.h>
#include <utils.h>
#include <test_base.h>

// #include "opflow_tests.h"
#include "test_acopf_utils.h"

PetscErrorCode ConstructSolutionVector(Vec *X, int num_copies);
PetscErrorCode ConstructReferenceJacobian(Mat *J, int num_copies);

/**
 * @brief Unit test driver for objective function
 * @see opflow/OpflowTests.hpp for kernel tested by this driver
 *
 * You can pass two options to the objectiveAcopf executatable through the
 * command line (implemented using PETSc options):
 *
 *    ~ -netfile <data_file> : Specifies the input data file to test against.
 * Default value is `/<exago_dir>/datafiles/case9/case9mod.m`. See directory
 * datafiles for other potential inputs.
 *
 *    ~ -num_copies <number> : Specifies the number of replications of the
 * network given through `-netfile`. If this is not set properly, test may fail
 *
 */
int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscBool flg;
  MPI_Comm comm = MPI_COMM_WORLD;

  char appname[] = "opflow";
  char help[] = "Unit tests for equality constraint Jacobians running opflow\n";

  /** Use `ExaGOLogSetLoggingFileName("opflow-logfile");` to log the output. */
  ierr = ExaGOInitialize(comm, &argc, &argv, appname, help);
  if (ierr) {
    fprintf(stderr, "Could not initialize ExaGO application %s.\n", appname);
    return ierr;
  }

  /* Get num_copies from command line */
  int num_copies = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_copies", &num_copies, &flg));

  // std::string netfile = "CECJ_unittestx" + std::to_string(num_copies) + ".m";
  std::string netfile = "CECJ_unittest1.m";

  Mat J_eq_ref;
  ConstructReferenceJacobian(&J_eq_ref, num_copies);
  Vec X;

  OPFLOW opflowtest;
  // exago::tests::TestOpflow test;

  /* Set up test opflow */
  PetscCall(OPFLOWCreate(PETSC_COMM_WORLD, &opflowtest));
  PetscCall(OPFLOWReadMatPowerData(opflowtest, netfile.c_str()));
  PetscCall(OPFLOWSetInitializationType(opflowtest, OPFLOWINIT_FROMFILE));
  PetscCall(OPFLOWSetUp(opflowtest));
  // PetscCall(OPFLOWGetSolution(opflowtest, &X));

  ConstructSolutionVector(&X, num_copies);

  // If we are using HIOP, need to convert X
  // The string lengths must be 65
  std::string modelname;
  std::string solvername;
  ierr = OPFLOWGetModel(opflowtest, &modelname);
  ierr = OPFLOWGetSolver(opflowtest, &solvername);

  int fail = 0;
  // if (solvername == "IPOPT") {
  Mat J_eq;
  Mat J_ineq = nullptr;
  PetscCall(MatDuplicate(J_eq_ref, MAT_SHARE_NONZERO_PATTERN, &J_eq));
  PetscCall(OPFLOWComputeConstraintJacobian(opflowtest, X, J_eq, J_ineq));

  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_ASCII_DENSE);
  PetscCall(MatView(J_eq_ref, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecView(X, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatView(J_eq, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatAXPY(J_eq, -1.0, J_eq_ref, SAME_NONZERO_PATTERN));
  PetscReal norm = 0.0;
  PetscCall(MatNorm(J_eq, NORM_INFINITY, &norm));
  std::cout << "Error norm: " << norm << std::endl;
  if (norm >= exago::tests::eps) {
    ++fail;
    ExaGOLog(
        EXAGO_LOG_INFO,
        "Error between Equality Constraint Jacobians ({}) exceeds tolerance {}",
        norm, exago::tests::eps);
  }
  // }
  // TODO: handle other solver types specially as necessary
  // else if (solvername == "HIOP") {
  // }
  // else if (solvername == "HIOPSPARSE") {
  // }
  // else {
  //     throw ExaGOError("Unsupported solver name: " + solvername);
  // }
  PetscCall(OPFLOWDestroy(&opflowtest));

  PetscCall(VecDestroy(&X));
  PetscCall(MatDestroy(&J_eq_ref));
  PetscCall(MatDestroy(&J_eq));
  // PetscCall(MatDestroy(&J_ineq));

  ExaGOFinalize();
  return fail;
}

PetscErrorCode ConstructSolutionVector(Vec *X, int num_copies) {
  PetscFunctionBeginUser;
  std::vector<double> x_base = {0, 2, 0, 2, 30, 2, 1.6, -2.2, 0, 2, 0, 2};
  int nvals_base = 12;
  int nvals = (nvals_base - 2) * num_copies + 2;
  std::vector<double> x;
  x.reserve(nvals);
  x.assign(begin(x_base), end(x_base));
  std::vector<int> is(nvals);
  std::iota(begin(is), end(is), 0);

  for (int n = 1; n < num_copies; ++n) {
    for (int i = 2; i < nvals_base; ++i) {
      x.push_back(x_base[i]);
    }
  }

  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, nvals, X));
  PetscCall(VecSetValues(*X, is.size(), is.data(), x.data(), ADD_VALUES));
  PetscCall(VecAssemblyBegin(*X));
  PetscCall(VecAssemblyEnd(*X));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ConstructReferenceJacobian(Mat *J, int num_copies) {
  PetscFunctionBeginUser;
  std::vector<int> i_base = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                             2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
                             4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7,
                             7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9};
  std::vector<int> j_base = {0, 1, 2,  3,  0, 1, 2,  3,  0, 1, 2,  3,  4, 5,
                             8, 9, 0,  1,  2, 3, 4,  5,  8, 9, 2,  3,  4, 5,
                             6, 2, 3,  4,  5, 7, 2,  3,  8, 9, 10, 11, 2, 3,
                             8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11};
  std::vector<double> v_base = {
      0.8,  0.8,  -0.8, -0.8, -1.6, -2.0, 1.6,  -0.4, -0.8, -0.8, 0.8,
      2.8,  0.8,  -0.2, -0.8, -0.8, 1.6,  -0.4, -3.6, -3.8, 0.4,  0.4,
      1.6,  -0.4, -0.8, 0.2,  0.8,  1.8,  -1.0, -0.4, -0.4, 0.4,  -2.0,
      -1.0, -0.8, -0.8, 1.6,  1.6,  -0.8, -0.8, 1.6,  -0.4, -3.2, -4.0,
      1.6,  -0.4, -0.8, -0.8, 0.8,  0.8,  1.6,  -0.4, -1.6, -2.0};

  int nrows_base = 10;
  int ncols_base = 12;
  int nrows = (nrows_base - 2) * num_copies + 2;
  int ncols = (ncols_base - 2) * num_copies + 2;

  std::vector<int> i_coo;
  i_coo.reserve(i_base.size() * num_copies);
  i_coo.assign(begin(i_base), end(i_base));
  std::vector<int> j_coo;
  j_coo.reserve(j_base.size() * num_copies);
  j_coo.assign(begin(j_base), end(j_base));
  std::vector<double> v_coo;
  v_coo.reserve(v_base.size() * num_copies);
  v_coo.assign(begin(v_base), end(v_base));

  for (int n = 1; n < num_copies; ++n) {
    auto row_start = n * (nrows_base - 2);
    auto col_start = n * (ncols_base - 2);
    for (std::size_t c = 0; c < v_base.size(); ++c) {
      i_coo.push_back(row_start + i_base[c]);
      j_coo.push_back(col_start + j_base[c]);
      v_coo.push_back(v_base[c]);
    }
  }

  PetscCall(MatCreateSeqAIJFromTriple(PETSC_COMM_WORLD, nrows, ncols,
                                      i_coo.data(), j_coo.data(), v_coo.data(),
                                      J, v_coo.size(), PETSC_FALSE));

  PetscFunctionReturn(PETSC_SUCCESS);
}
