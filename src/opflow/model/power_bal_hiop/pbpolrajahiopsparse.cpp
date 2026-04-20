#include <exago_config.h>

#if defined(EXAGO_ENABLE_RAJA)
#if defined(EXAGO_ENABLE_HIOP_SPARSE)

#include <map>

#include <private/opflowimpl.h>
#include "pbpolrajahiopsparsekernels.hpp"

/**
 * @brief `extern` initial guess function from PBPOL model.
 *
 * Initialization is done on the host through this function. Copying over values
 * to the device is done in OPFLOWSetInitialGuessArray_PBPOLRAJAHIOPSPARSE
 */
extern PetscErrorCode OPFLOWSetInitialGuess_PBPOL(OPFLOW, Vec, Vec);

/**
 * @brief Set the initial guess for the PBPOL model with Raja and HiOp sparse.
 *
 * This function sets the initial guess by calling the same function from
 * the PBPOL model.
 */
PetscErrorCode OPFLOWSetInitialGuess_PBPOLRAJAHIOPSPARSE(OPFLOW opflow, Vec X,
                                                         Vec Lambda) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = OPFLOWSetInitialGuess_PBPOL(opflow, X, Lambda);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// @brief `extern` constraint bounds function from PBPOL model.
extern PetscErrorCode OPFLOWSetConstraintBounds_PBPOL(OPFLOW, Vec, Vec);

/**
 * @brief Set the constraint bounds for the PBPOLHIOPSPARSE model.
 *
 * The constraint bounds are also calculated on the host by matching function
 * from PBPOL model.
 */
PetscErrorCode OPFLOWSetConstraintBounds_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                                             Vec Gl, Vec Gu) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = OPFLOWSetConstraintBounds_PBPOL(opflow, Gl, Gu);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// @brief `extern` variable bounds function from PBPOL model.
extern PetscErrorCode OPFLOWSetVariableBounds_PBPOL(OPFLOW, Vec, Vec);

/**
 * @brief Set the variable bounds for the PBPOLHIOPSPARSE model.
 *
 * The variable bounds are also calculated on the host by matching function
 * from PBPOL model.
 */
PetscErrorCode OPFLOWSetVariableBounds_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                                           Vec Xl, Vec Xu) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = OPFLOWSetVariableBounds_PBPOL(opflow, Xl, Xu);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/**
 * @brief Store PBPOLRAJAHIOPSPARSE solution to the PS structure.
 *
 */
PetscErrorCode OPFLOWSolutionToPS_PBPOLRAJAHIOPSPARSE(OPFLOW opflow) {
  PetscErrorCode ierr;
  PS ps = (PS)opflow->ps;
  PetscInt i, k;
  Vec X, Lambda;
  PSBUS bus;
  PSGEN gen;
  PSLOAD load;
  PSLINE line;
  const PetscScalar *x, *lambda, *lambdae, *lambdai;
  PetscInt loc, gloc = 0;
  PetscScalar Gff, Bff, Gft, Bft, Gtf, Btf, Gtt, Btt;
  PetscScalar Vmf, Vmt, thetaf, thetat, thetaft, thetatf;
  PetscScalar Pf, Qf, Pt, Qt;
  PSBUS busf, bust;
  const PSBUS *connbuses;
  PetscInt xlocf, xloct;

  PetscFunctionBegin;

  ierr = OPFLOWGetSolution(opflow, &X);
  CHKERRQ(ierr);
  ierr = OPFLOWGetConstraintMultipliers(opflow, &Lambda);
  CHKERRQ(ierr);

  ierr = VecGetArrayRead(X, &x);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(Lambda, &lambda);
  CHKERRQ(ierr);
  lambdae = lambda;
  if (opflow->Nconineq) {
    lambdai = lambdae + opflow->nconeq;
  }

  for (i = 0; i < ps->nbus; i++) {
    bus = &ps->bus[i];

    loc = bus->startxVloc;

    bus->va = x[loc];
    bus->vm = x[loc + 1];

    gloc = bus->starteqloc;
    bus->mult_pmis = lambdae[gloc];
    bus->mult_qmis = lambdae[gloc + 1];

    if (opflow->include_powerimbalance_variables) {
      loc = bus->startxpimbloc;
      bus->pimb = x[loc] - x[loc + 1];
      bus->qimb = x[loc + 2] - x[loc + 3];
    }

    for (k = 0; k < bus->ngen; k++) {
      ierr = PSBUSGetGen(bus, k, &gen);
      CHKERRQ(ierr);
      if (!gen->status) {
        gen->pg = gen->qg = 0.0;
        continue;
      }
      loc = gen->startxpowloc;

      gen->pg = x[loc];
      gen->qg = x[loc + 1];

      if (opflow->has_gensetpoint && !gen->isrenewable) {
        gloc += gen->nconeq;
      }
    }

    if (opflow->include_loadloss_variables) {
      for (k = 0; k < bus->nload; k++) {
        ierr = PSBUSGetLoad(bus, k, &load);
        CHKERRQ(ierr);
        loc = load->startxloadlossloc;
        load->pl_loss = x[loc];
        load->ql_loss = x[loc + 1];
      }
    }
  }

  if (!opflow->ignore_lineflow_constraints) {
    for (i = 0; i < ps->nline; i++) {
      line = &ps->line[i];
      if (!line->status) {
        line->mult_sf = line->mult_st = 0.0;
        continue;
      }

      Gff = line->yff[0];
      Bff = line->yff[1];
      Gft = line->yft[0];
      Bft = line->yft[1];
      Gtf = line->ytf[0];
      Btf = line->ytf[1];
      Gtt = line->ytt[0];
      Btt = line->ytt[1];

      ierr = PSLINEGetConnectedBuses(line, &connbuses);
      CHKERRQ(ierr);
      busf = connbuses[0];
      bust = connbuses[1];

      xlocf = busf->startxVloc;
      xloct = bust->startxVloc;

      thetaf = x[xlocf];
      Vmf = x[xlocf + 1];
      thetat = x[xloct];
      Vmt = x[xloct + 1];
      thetaft = thetaf - thetat;
      thetatf = thetat - thetaf;

      Pf = Gff * Vmf * Vmf +
           Vmf * Vmt * (Gft * cos(thetaft) + Bft * sin(thetaft));
      Qf = -Bff * Vmf * Vmf +
           Vmf * Vmt * (-Bft * cos(thetaft) + Gft * sin(thetaft));

      Pt = Gtt * Vmt * Vmt +
           Vmt * Vmf * (Gtf * cos(thetatf) + Btf * sin(thetatf));
      Qt = -Btt * Vmt * Vmt +
           Vmt * Vmf * (-Btf * cos(thetatf) + Gtf * sin(thetatf));

      line->pf = Pf;
      line->qf = Qf;
      line->pt = Pt;
      line->qt = Qt;
      line->sf = PetscSqrtScalar(Pf * Pf + Qf * Qf);
      line->st = PetscSqrtScalar(Pt * Pt + Qt * Qt);

      if (line->rateA > 1e5) {
        line->mult_sf = line->mult_st = 0.0;
      } else {
        gloc = line->startineqloc;
        line->mult_sf = lambdai[gloc];
        line->mult_st = lambdai[gloc + 1];
      }
    }
  }

  ierr = VecRestoreArrayRead(X, &x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Lambda, &lambda);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Reuse PBPOL model set up for obtaining locations */
/// @brief `extern` set up function from PBPOL model.
extern PetscErrorCode OPFLOWModelSetUp_PBPOL(OPFLOW);

/** @brief Set up the PBPOLRAJAHIOPSPARSE model.
 *
 * This function initializes the host objects for the PBPOLRAJAHIOPSPARSE
 * model by calling the corresponding function from the PBPOL model, and then
 * allocates structures with device data, defined in `paramsrajahiop.cpp`.
 */
PetscErrorCode OPFLOWModelSetUp_PBPOLRAJAHIOPSPARSE(OPFLOW opflow) {
  PetscErrorCode ierr;

  // This struct is defined in `paramsrajahiop.h`. It inherits from
  // _p_FormPBPOLRAJAHIOP, which is unwise mixture of C and C++ approaches.
  PbpolModelRajaHiop *pbpolrajahiopsparse =
      reinterpret_cast<PbpolModelRajaHiop *>(opflow->model);

  PetscFunctionBegin;

  ierr = OPFLOWModelSetUp_PBPOL(opflow);
  CHKERRQ(ierr);

  ierr = pbpolrajahiopsparse->busparams.allocate(opflow);
  ierr = pbpolrajahiopsparse->genparams.allocate(opflow);
  ierr = pbpolrajahiopsparse->lineparams.allocate(opflow);
  ierr = pbpolrajahiopsparse->loadparams.allocate(opflow);

  BUSParamsRajaHiop *busparams = &pbpolrajahiopsparse->busparams;
  GENParamsRajaHiop *genparams = &pbpolrajahiopsparse->genparams;
  LOADParamsRajaHiop *loadparams = &pbpolrajahiopsparse->loadparams;
  LINEParamsRajaHiop *lineparams = &pbpolrajahiopsparse->lineparams;

  /* Compute the number of nonzeros in equality constraint Jacobian
   * and populate flat-array indices for the GPU kernel. */
  int nnz_eqjacsp = 0, nnz_ineqjac = 0, nnz_hess = 0;

  PS ps = (PS)opflow->ps;
  int geni = 0, loadi = 0;

  for (int ibus = 0; ibus < ps->nbus; ++ibus) {
    PSBUS bus = &(ps->bus[ibus]);

    /* P-row: bus self-admittance (theta, Vm) */
    busparams->eqjacsp_selfidx[2 * ibus] = nnz_eqjacsp;
    nnz_eqjacsp += 2;

    if (bus->ide == ISOLATED_BUS) {
      /* Q-row: bus self-admittance for isolated bus */
      busparams->eqjacsp_selfidx[2 * ibus + 1] = nnz_eqjacsp;
      nnz_eqjacsp += 2;
      continue;
    }

    /* P-row: power imbalance variables */
    if (opflow->include_powerimbalance_variables) {
      busparams->jacsp_idx[ibus] = nnz_eqjacsp;
      nnz_eqjacsp += 2;
    }

    /* P-row: generator Pg entries (-1 per active gen) */
    int gi = 0;
    for (int k = 0; k < bus->ngen; k++) {
      PSGEN gen;
      ierr = PSBUSGetGen(bus, k, &gen);
      CHKERRQ(ierr);
      if (!gen->status)
        continue;
      genparams->eqjacspbus_idx[geni + gi] = nnz_eqjacsp;
      nnz_eqjacsp += 1;
      gi++;
    }

    /* P-row: load loss entries (-1 per load) */
    if (opflow->include_loadloss_variables) {
      for (int k = 0; k < bus->nload; k++) {
        PSLOAD load;
        ierr = PSBUSGetLoad(bus, k, &load);
        CHKERRQ(ierr);
        loadparams->jacsp_idx[loadi + k] = nnz_eqjacsp;
        nnz_eqjacsp += 1;
      }
    }

    /* Q-row: bus self-admittance (theta, Vm) */
    busparams->eqjacsp_selfidx[2 * ibus + 1] = nnz_eqjacsp;
    nnz_eqjacsp += 2;

    /* Q-row: power imbalance variables */
    if (opflow->include_powerimbalance_variables) {
      busparams->jacsq_idx[ibus] = nnz_eqjacsp;
      nnz_eqjacsp += 2;
    }

    /* Q-row: generator Qg entries (-1 per active gen) */
    gi = 0;
    for (int k = 0; k < bus->ngen; k++) {
      PSGEN gen;
      ierr = PSBUSGetGen(bus, k, &gen);
      CHKERRQ(ierr);
      if (!gen->status)
        continue;
      genparams->eqjacsqbus_idx[geni + gi] = nnz_eqjacsp;
      nnz_eqjacsp += 1;
      gi++;
    }

    /* Q-row: load loss entries (-1 per load) */
    if (opflow->include_loadloss_variables) {
      for (int k = 0; k < bus->nload; k++) {
        PSLOAD load;
        ierr = PSBUSGetLoad(bus, k, &load);
        CHKERRQ(ierr);
        loadparams->jacsq_idx[loadi + k] = nnz_eqjacsp;
        nnz_eqjacsp += 1;
      }
    }

    geni += bus->ngenON;
    loadi += bus->nload;
  }

  /* Line off-diagonal entries: 8 per unique (from_bus, to_bus) pair.
     Parallel lines (same bus pair) share the same off-diagonal positions
     and accumulate via atomicAdd in the GPU kernel. */
  int linei = 0;
  std::map<std::pair<int, int>, int> buspair_to_offdiag;
  for (int iline = 0; iline < ps->nline; ++iline) {
    PSLINE line = &(ps->line[iline]);
    if (!line->status)
      continue;

    if (!line->isdcline) {
      const PSBUS *connbuses;
      ierr = PSLINEGetConnectedBuses(line, &connbuses);
      CHKERRQ(ierr);
      PSBUS busf = connbuses[0];
      PSBUS bust = connbuses[1];
      int busidxf = (int)(busf - ps->bus);
      int busidxt = (int)(bust - ps->bus);

      lineparams->eqjacsp_diag_idx[4 * linei + 0] =
          busparams->eqjacsp_selfidx[2 * busidxf];
      lineparams->eqjacsp_diag_idx[4 * linei + 1] =
          busparams->eqjacsp_selfidx[2 * busidxf + 1];
      lineparams->eqjacsp_diag_idx[4 * linei + 2] =
          busparams->eqjacsp_selfidx[2 * busidxt];
      lineparams->eqjacsp_diag_idx[4 * linei + 3] =
          busparams->eqjacsp_selfidx[2 * busidxt + 1];

      auto key = std::make_pair(std::min(busidxf, busidxt),
                                std::max(busidxf, busidxt));
      auto it = buspair_to_offdiag.find(key);
      if (it != buspair_to_offdiag.end()) {
        lineparams->eqjacsp_idx[linei] = it->second;
      } else {
        lineparams->eqjacsp_idx[linei] = nnz_eqjacsp;
        buspair_to_offdiag[key] = nnz_eqjacsp;
        nnz_eqjacsp += 8;
      }
    }

    linei++;
  }

  /* Generator set-point equality constraint entries */
  if (opflow->has_gensetpoint) {
    geni = 0;
    for (int ibus = 0; ibus < ps->nbus; ++ibus) {
      PSBUS bus = &(ps->bus[ibus]);
      int gi = 0;
      for (int k = 0; k < bus->ngen; k++) {
        PSGEN gen;
        ierr = PSBUSGetGen(bus, k, &gen);
        CHKERRQ(ierr);
        if (!gen->status)
          continue;
        if (!gen->isrenewable) {
          genparams->eqjacspgen_idx[geni + gi] = nnz_eqjacsp;
          nnz_eqjacsp += 4;
        }
        gi++;
      }
      geni += bus->ngenON;
    }
  }

  if (opflow->has_gensetpoint) {
    for (int ibus = 0; ibus < ps->nbus; ++ibus) {
      PSBUS bus = &(ps->bus[ibus]);
      for (int bgen = 0; bgen < bus->ngen; ++bgen) {
        PSGEN gen;
        ierr = PSBUSGetGen(bus, bgen, &gen);
        CHKERRQ(ierr);

        if (!gen->status)
          continue;

        nnz_ineqjac += 6;
      }
    }
  }

  if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
    for (int ibus = 0; ibus < ps->nbus; ++ibus) {
      PSBUS bus = &(ps->bus[ibus]);
      if (bus->ide == PV_BUS || bus->ide == REF_BUS) {
        nnz_ineqjac += 2;
      }
    }
  }

  for (int iline = 0; iline < opflow->nlinesmon; ++iline) {
    nnz_ineqjac += 8;
  }

  for (int ibus = 0; ibus < ps->nbus; ++ibus) {
    // reserve 2 real and 2 reactive entries for each bus
    // 3 upper triangular
    nnz_hess += 3;

    if (opflow->include_powerimbalance_variables) {
      nnz_hess += 2;
    }
  }

  for (int i = 0; i < ps->ngen; ++i) {
    PSGEN gen = &(ps->gen[i]);

    if (!gen->status)
      continue;

    nnz_hess += 2;

    if (opflow->has_gensetpoint) {
      if (gen->isrenewable)
        continue;

      // later ...
      // if (opflow->use_agc) {
      //   nnz_hess += 5;
      // }
    }
    if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
      nnz_hess += 2;
    }
  }

  for (int iline = 0; iline < ps->nline; ++iline) {
    PSLINE line = &(ps->line[iline]);

    if (!line->status)
      continue;

    // 3 diagonal entries for on the from-bus rows (already defined)
    // 3 diagonal entries for on the to-bus rows (already defined)
    // 4 off-diagonal entries in upper part
    nnz_hess += 4;
  }

  if (opflow->include_loadloss_variables) {
    for (int iload = 0; iload < ps->nload; ++iload) {
      nnz_hess += 2;
    }
  }

  opflow->nnz_eqjacsp = nnz_eqjacsp;
  opflow->nnz_ineqjacsp = nnz_ineqjac;
  opflow->nnz_hesssp = nnz_hess;

  ierr = busparams->copy(opflow);
  ierr = genparams->copy(opflow);
  ierr = lineparams->copy(opflow);
  ierr = loadparams->copy(opflow);

  PetscFunctionReturn(0);
}

/**
 * @brief Destructor for the PBPOLRAJAHIOPSPARSE model.
 *
 */
PetscErrorCode OPFLOWModelDestroy_PBPOLRAJAHIOPSPARSE(OPFLOW opflow) {
  PbpolModelRajaHiop *pbpolrajahiopsparse =
      reinterpret_cast<PbpolModelRajaHiop *>(opflow->model);

  PetscFunctionBegin;
  pbpolrajahiopsparse->destroy(opflow);
  delete pbpolrajahiopsparse;
  pbpolrajahiopsparse = nullptr;

  PetscFunctionReturn(0);
}

/* reuse numvariables and numconstraints functions from PBPOL model */
extern PetscErrorCode OPFLOWModelSetNumVariables_PBPOL(OPFLOW, PetscInt *,
                                                       PetscInt *, PetscInt *);
extern PetscErrorCode OPFLOWModelSetNumConstraints_PBPOL(OPFLOW, PetscInt *,
                                                         PetscInt *, PetscInt *,
                                                         PetscInt *);
extern PetscErrorCode OPFLOWComputeEqualityConstraintJacobian_PBPOL(OPFLOW, Vec,
                                                                    Mat);
extern PetscErrorCode OPFLOWComputeInequalityConstraintJacobian_PBPOL(OPFLOW,
                                                                      Vec, Mat);
extern PetscErrorCode OPFLOWComputeHessian_PBPOL(OPFLOW, Vec, Vec, Vec, Mat);

extern PetscErrorCode OPFLOWSolutionCallback_PBPOLRAJAHIOPSPARSE(
    OPFLOW, const double *, const double *, const double *, const double *,
    const double *, double);

/**
 * Empty stub for the equality constraint Jacobian in the RAJA sparse model.
 * Will be replaced with a full RAJA kernel implementation.
 */
PetscErrorCode
OPFLOWComputeEqualityConstraintJacobian_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                                             Vec X, Mat Je) {
  (void)opflow;
  (void)X;
  (void)Je;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/**
 * @brief Constructor for the PBPOLRAJAHIOPSPARSE model.
 *
 * This function creates a new PBPOLRAJAHIOPSPARSE model and sets pointers
 * to the model's methods implementations.
 *
 * @param opflow The pointer to the object to instantiate.
 * @return PetscErrorCode indicating success or failure.
 */
PetscErrorCode OPFLOWModelCreate_PBPOLRAJAHIOPSPARSE(OPFLOW opflow) {

  PetscFunctionBegin;

  // This may be confusing: The class PbpolModelRajaHiop is a sparse model
  // using HiOp solver, but is implemented in file with HiOp mixed dense-sparse
  // model implementation.
  PbpolModelRajaHiop *pbpolrajahiopsparse = new PbpolModelRajaHiop();

  opflow->model = pbpolrajahiopsparse;

  /* PBPOLRAJAHIOPSPARSE models only support VARIABLE_WITHIN_BOUNDS
   * opflow->genbusvoltagetype
   */
  opflow->genbusvoltagetype = VARIABLE_WITHIN_BOUNDS;

  opflow->spdnordering = PETSC_FALSE;

  /* Inherit Ops */
  opflow->modelops.destroy = OPFLOWModelDestroy_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setnumvariables = OPFLOWModelSetNumVariables_PBPOL;
  opflow->modelops.setnumconstraints = OPFLOWModelSetNumConstraints_PBPOL;
  opflow->modelops.setvariablebounds =
      OPFLOWSetVariableBounds_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setvariableboundsarray =
      OPFLOWSetVariableBoundsArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setconstraintbounds =
      OPFLOWSetConstraintBounds_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setconstraintboundsarray =
      OPFLOWSetConstraintBoundsArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setinitialguess = OPFLOWSetInitialGuess_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setinitialguessarray =
      OPFLOWSetInitialGuessArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computeequalityconstraintsarray =
      OPFLOWComputeEqualityConstraintsArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computeinequalityconstraintsarray =
      OPFLOWComputeInequalityConstraintsArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computeobjectivearray =
      OPFLOWComputeObjectiveArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computegradientarray =
      OPFLOWComputeGradientArray_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.solutiontops = OPFLOWSolutionToPS_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.setup = OPFLOWModelSetUp_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computeequalityconstraintjacobian =
      OPFLOWComputeEqualityConstraintJacobian_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computesparseequalityconstraintjacobianhiop =
      OPFLOWComputeSparseEqualityConstraintJacobian_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computeinequalityconstraintjacobian =
      OPFLOWComputeInequalityConstraintJacobian_PBPOL;
  opflow->modelops.computesparseinequalityconstraintjacobianhiop =
      OPFLOWComputeSparseInequalityConstraintJacobian_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.computehessian = OPFLOWComputeHessian_PBPOL;
  opflow->modelops.computesparsehessianhiop =
      OPFLOWComputeSparseHessian_PBPOLRAJAHIOPSPARSE;
  opflow->modelops.solutioncallbackhiop =
      OPFLOWSolutionCallback_PBPOLRAJAHIOPSPARSE;

  PetscFunctionReturn(0);
}

#endif
#endif
