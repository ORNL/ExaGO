#include <fstream>
#include <iostream>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opflow.h>
#include <utils.h>
#include <test_base.h>

inline constexpr double PI = 3.14159265358979323846;

PetscErrorCode ConstructSolutionVector(Vec *X, int num_copies);
PetscErrorCode ConstructReferenceJacobian(Mat *J, int num_copies,
                                          const std::string &reffile);

/**
 * @brief Unit test driver for the inequality constraint Jacobian
 *
 * Computes the inequality constraint Jacobian of the network at a fixed
 * solution vector and compares it with the reference Jacobian of the 5-bus
 * network, replicated `num_copies` times.
 *
 * Options (implemented using PETSc options):
 *
 *    ~ -netfile <data_file> : Network file, e.g. `CICJ_unittestx<N>.m` or
 * `CICJ_nolinelimits_unittestx<N>.m`.
 *
 *    ~ -validation <csv_file> : Reference Jacobian of the 5-bus network, e.g.
 * `cicj.csv` or `cicj_nolinelimits.csv`.
 *
 *    ~ -num_copies <number> : Number of copies N of the 5-bus network in the
 * network file (default 1). If this is not set properly, the test fails.
 *
 */
int main(int argc, char **argv) {
  PetscBool flg;

  char appname[] = "opflow";
  char help[] =
      "Unit tests for inequality constraint Jacobians running opflow\n";

  /** Use `ExaGOLogSetLoggingFileName("opflow-logfile");` to log the output. */
  PetscErrorCode ierr =
      ExaGOInitialize(MPI_COMM_WORLD, &argc, &argv, appname, help);
  if (ierr) {
    fprintf(stderr, "Could not initialize ExaGO application %s.\n", appname);
    return ierr;
  }

  /* Get num_copies from command line */
  int num_copies = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_copies", &num_copies, &flg));

  /* Get network and reference Jacobian files from command line */
  char netfile[PETSC_MAX_PATH_LEN] = "";
  char reffile[PETSC_MAX_PATH_LEN] = "";
  PetscCall(PetscOptionsGetString(NULL, NULL, "-netfile", netfile,
                                  PETSC_MAX_PATH_LEN, &flg));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-validation", reffile,
                                  PETSC_MAX_PATH_LEN, &flg));

  Mat J_ineq_ref;
  PetscCall(ConstructReferenceJacobian(&J_ineq_ref, num_copies, reffile));
  Vec X;

  OPFLOW opflowtest;

  /* Set up test opflow */
  PetscCall(OPFLOWCreate(PETSC_COMM_WORLD, &opflowtest));
  PetscCall(OPFLOWReadMatPowerData(opflowtest, netfile));
  PetscCall(OPFLOWSetGenBusVoltageType(opflowtest, FIXED_WITHIN_QBOUNDS));
  PetscCall(OPFLOWSetUp(opflowtest));

  PetscCall(ConstructSolutionVector(&X, num_copies));

  std::string solvername;
  PetscCall(OPFLOWGetSolver(opflowtest, &solvername));

  int fail = 0;
  if (solvername == "IPOPT" || solvername == "HIOPSPARSE") {
    Mat J_eq;
    Mat J_ineq;
    PetscCall(OPFLOWGetConstraintJacobian(opflowtest, &J_eq, &J_ineq));
    PetscCall(OPFLOWComputeConstraintJacobian(opflowtest, X, J_eq, J_ineq));

    PetscCall(MatAXPY(J_ineq, -1.0, J_ineq_ref, UNKNOWN_NONZERO_PATTERN));
    PetscReal norm = 0.0;
    PetscCall(MatNorm(J_ineq, NORM_INFINITY, &norm));
    std::cout << "Error norm: " << norm << std::endl;
    if (norm >= exago::tests::eps) {
      ++fail;
      ExaGOLog(EXAGO_LOG_INFO,
               "Error between Inequality Constraint Jacobians ({}) exceeds "
               "tolerance {}",
               norm, exago::tests::eps);
    }
  } else {
    ExaGOLog(EXAGO_LOG_INFO, "Skipping test for unsupported solver {}",
             solvername);
    fail = exago::tests::SKIP_TEST;
  }

  PetscCall(OPFLOWDestroy(&opflowtest));

  PetscCall(VecDestroy(&X));
  PetscCall(MatDestroy(&J_ineq_ref));

  ExaGOFinalize();
  return fail;
}

PetscErrorCode ConstructSolutionVector(Vec *X, int num_copies) {
  PetscFunctionBeginUser;

  std::vector<PetscReal> x_base = {0, 2, 0, 2, 30 * PI / 180.0, 2, 1.6, -2.2,
                                   0, 2, 0, 2};
  int nvals_base = x_base.size();
  int nvals = (nvals_base - 2) * num_copies + 2;
  std::vector<PetscReal> x;
  x.reserve(nvals);
  x.assign(begin(x_base), end(x_base));
  std::vector<PetscInt> is(nvals);
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

PetscErrorCode ConstructReferenceJacobian(Mat *J, int num_copies,
                                          const std::string &reffile) {
  PetscFunctionBeginUser;

  // Read base Jacobian from file
  std::ifstream ifs(reffile);
  if (!ifs) {
    throw ExaGOError("Unable to open file: " + reffile);
  }
  std::string line;
  int nrows_base, ncols_base;
  std::vector<PetscInt> i_base;
  std::vector<PetscInt> j_base;
  std::vector<PetscReal> v_base;
  // Read matrix dimensions
  std::getline(ifs, line);
  std::istringstream iss(line);
  std::string dimstr;
  std::getline(iss, dimstr, ',');
  std::istringstream(dimstr) >> nrows_base;
  std::getline(iss, dimstr, ',');
  std::istringstream(dimstr) >> ncols_base;
  // Read triples
  std::string estr;
  while (std::getline(ifs, line)) {
    PetscInt i, j;
    PetscReal v;
    iss = std::istringstream(line);
    std::getline(iss, estr, ',');
    std::istringstream(estr) >> i;
    std::getline(iss, estr, ',');
    std::istringstream(estr) >> j;
    std::getline(iss, estr, ',');
    std::istringstream(estr) >> v;
    i_base.push_back(i - 1);
    j_base.push_back(j - 1);
    v_base.push_back(v);
  }

  // The first 2 rows are generator bus constraints. OPFLOW numbers all bus
  // constraints before all line flow constraints.
  int nrows_top = 2 * num_copies;
  int nrows = nrows_base * num_copies;
  int ncols = (ncols_base - 2) * num_copies + 2;

  std::vector<PetscInt> i_coo;
  std::vector<PetscInt> j_coo;
  std::vector<PetscScalar> v_coo;

  std::size_t ncoo = i_base.size() * num_copies;
  i_coo.reserve(ncoo);
  j_coo.reserve(ncoo);
  v_coo.reserve(ncoo);

  for (int n = 0; n < num_copies; ++n) {
    auto row_start_top = n * 2;
    auto row_start = nrows_top + n * (nrows_base - 2);
    auto col_start = n * (ncols_base - 2);
    for (std::size_t c = 0; c < v_base.size(); ++c) {
      auto i = i_base[c];
      if (i < 2) {
        i_coo.push_back(row_start_top + i);
      } else {
        i_coo.push_back(row_start + i - 2);
      }
      j_coo.push_back(col_start + j_base[c]);
      v_coo.push_back(v_base[c]);
    }
  }

  PetscCall(MatCreateSeqAIJFromTriple(PETSC_COMM_WORLD, nrows, ncols,
                                      i_coo.data(), j_coo.data(), v_coo.data(),
                                      J, v_coo.size(), PETSC_FALSE));

  PetscFunctionReturn(PETSC_SUCCESS);
}
