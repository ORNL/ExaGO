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

/** @brief Helper function to ensure Hessian entries are not duplicated
 *
 */
static inline int
count_entry(std::map<std::pair<int, int>, int> &existing_pairs, int r, int c,
            int &nnz_hesssp) {
  if (r > c)
    std::swap(r, c);
  const auto key = std::make_pair(r, c);

  auto it = existing_pairs.find(key);
  if (it == existing_pairs.end()) {
    const int idx = nnz_hesssp;
    existing_pairs[key] = idx;
    nnz_hesssp++;
    return idx;
  }
  return it->second;
}

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

  PS ps = opflow->ps;
  PSBUS bus;
  PSGEN gen;
  PSLINE line;
  PetscInt i, k;

  /* Store the AGC variable index (scalar) */
  if (opflow->use_agc) {
    pbpolrajahiopsparse->agc_xidx = ps->startxloc;
  } else {
    pbpolrajahiopsparse->agc_xidx = -1;
  }

  /* Initialize the number of nonzeros to 0 */
  int nnz_eqjacsp = 0, nnz_ineqjacsp = 0, nnz_hesssp = 0;

  /* ---- Equality constraint Jacobian nnz counting ---- */
  {
    int geni_eq = 0, loadi_eq = 0;
    for (int ibus = 0; ibus < ps->nbus; ++ibus) {
      PSBUS bus_eq = &(ps->bus[ibus]);

      busparams->eqjacsp_idx[2 * ibus] = nnz_eqjacsp;
      nnz_eqjacsp += 2;

      if (bus_eq->ide == ISOLATED_BUS) {
        busparams->eqjacsp_idx[2 * ibus + 1] = nnz_eqjacsp;
        nnz_eqjacsp += 2;
        continue;
      }

      if (opflow->include_powerimbalance_variables) {
        busparams->jacsp_idx[ibus] = nnz_eqjacsp;
        nnz_eqjacsp += 2;
      }

      int gi_eq = 0;
      for (int kk = 0; kk < bus_eq->ngen; kk++) {
        PSGEN gen_eq;
        ierr = PSBUSGetGen(bus_eq, kk, &gen_eq);
        CHKERRQ(ierr);
        if (!gen_eq->status)
          continue;
        genparams->eqjacspbus_idx[geni_eq + gi_eq] = nnz_eqjacsp;
        nnz_eqjacsp += 1;
        gi_eq++;
      }

      if (opflow->include_loadloss_variables) {
        for (int kk = 0; kk < bus_eq->nload; kk++) {
          PSLOAD load_eq;
          ierr = PSBUSGetLoad(bus_eq, kk, &load_eq);
          CHKERRQ(ierr);
          loadparams->jacsp_idx[loadi_eq + kk] = nnz_eqjacsp;
          nnz_eqjacsp += 1;
        }
      }

      busparams->eqjacsp_idx[2 * ibus + 1] = nnz_eqjacsp;
      nnz_eqjacsp += 2;

      if (opflow->include_powerimbalance_variables) {
        busparams->jacsq_idx[ibus] = nnz_eqjacsp;
        nnz_eqjacsp += 2;
      }

      gi_eq = 0;
      for (int kk = 0; kk < bus_eq->ngen; kk++) {
        PSGEN gen_eq;
        ierr = PSBUSGetGen(bus_eq, kk, &gen_eq);
        CHKERRQ(ierr);
        if (!gen_eq->status)
          continue;
        genparams->eqjacsqbus_idx[geni_eq + gi_eq] = nnz_eqjacsp;
        nnz_eqjacsp += 1;
        gi_eq++;
      }

      if (opflow->include_loadloss_variables) {
        for (int kk = 0; kk < bus_eq->nload; kk++) {
          PSLOAD load_eq;
          ierr = PSBUSGetLoad(bus_eq, kk, &load_eq);
          CHKERRQ(ierr);
          loadparams->jacsq_idx[loadi_eq + kk] = nnz_eqjacsp;
          nnz_eqjacsp += 1;
        }
      }

      geni_eq += bus_eq->ngenON;
      loadi_eq += bus_eq->nload;
    }

    int iline_eq = 0;
    std::map<std::pair<int, int>, int> buspair_to_offdiag;
    for (int iline = 0; iline < ps->nline; ++iline) {
      PSLINE line_eq = &(ps->line[iline]);
      if (!line_eq->status)
        continue;
      if (!line_eq->isdcline) {
        const PSBUS *connbuses_eq;
        ierr = PSLINEGetConnectedBuses(line_eq, &connbuses_eq);
        CHKERRQ(ierr);
        int busidxf = (int)(connbuses_eq[0] - ps->bus);
        int busidxt = (int)(connbuses_eq[1] - ps->bus);

        lineparams->eqjacsp_diag_idx[4 * iline_eq + 0] =
            busparams->eqjacsp_idx[2 * busidxf];
        lineparams->eqjacsp_diag_idx[4 * iline_eq + 1] =
            busparams->eqjacsp_idx[2 * busidxf + 1];
        lineparams->eqjacsp_diag_idx[4 * iline_eq + 2] =
            busparams->eqjacsp_idx[2 * busidxt];
        lineparams->eqjacsp_diag_idx[4 * iline_eq + 3] =
            busparams->eqjacsp_idx[2 * busidxt + 1];

        auto key = std::make_pair(std::min(busidxf, busidxt),
                                  std::max(busidxf, busidxt));
        auto it = buspair_to_offdiag.find(key);
        if (it != buspair_to_offdiag.end()) {
          lineparams->eqjacsp_idx[iline_eq] = it->second;
        } else {
          lineparams->eqjacsp_idx[iline_eq] = nnz_eqjacsp;
          buspair_to_offdiag[key] = nnz_eqjacsp;
          nnz_eqjacsp += 8;
        }
      }
      iline_eq++;
    }

    if (opflow->has_gensetpoint) {
      geni_eq = 0;
      for (int ibus = 0; ibus < ps->nbus; ++ibus) {
        PSBUS bus_eq = &(ps->bus[ibus]);
        int gi_eq = 0;
        for (int kk = 0; kk < bus_eq->ngen; kk++) {
          PSGEN gen_eq;
          ierr = PSBUSGetGen(bus_eq, kk, &gen_eq);
          CHKERRQ(ierr);
          if (!gen_eq->status)
            continue;
          if (!gen_eq->isrenewable) {
            genparams->eqjacspgen_idx[geni_eq + gi_eq] = nnz_eqjacsp;
            nnz_eqjacsp += 4;
          }
          gi_eq++;
        }
        geni_eq += bus_eq->ngenON;
      }
    }
  }

  /* ---- Inequality constraint Jacobian nnz counting ---- */
  {
    int geni = 0, gi;
    for (i = 0; i < ps->nbus; i++) {
      bus = &ps->bus[i];

      /* Bus voltage-Q-bounds constraints (FIXED_WITHIN_QBOUNDS) */
      if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
        if (bus->ide == PV_BUS || bus->ide == REF_BUS) {
          busparams->ineqjacsp_idx[i] = nnz_ineqjacsp;
          /* 2 rows, each with ngenON + 1 entries (one per gen Qg + one for V)
           */
          nnz_ineqjacsp += 2 * (bus->ngenON + 1);
        }
      }

      /* KS: Generator set-point constraints */
      gi = 0;
      if (opflow->has_gensetpoint) {
        for (k = 0; k < bus->ngen; k++) {
          ierr = PSBUSGetGen(bus, k, &gen);
          CHKERRQ(ierr);
          if (!gen->status)
            continue;
          if (!gen->isrenewable) {
            genparams->ineqjacspgen_idx[geni + gi] = nnz_ineqjacsp;
            if (opflow->use_agc) {
              nnz_ineqjacsp += 6; /* 2 rows x 3 entries (Pg, delPg, delP) */
            }
          }
          gi++;
        }
      }

      geni += bus->ngenON;
    }

    /* Line flow constraints */
    int linej = 0;
    for (i = 0; i < ps->nline; i++) {
      line = &ps->line[i];
      if (!line->status)
        continue;
      if (line->isdcline)
        continue;

      if (linej < opflow->nlinesmon && opflow->linesmon[linej] == i) {
        lineparams->ineqjacsp_idx[linej] = nnz_ineqjacsp;
        /* 2 rows x 4 entries (thetaf, Vmf, thetat, Vmt) */
        int entries_per_line = 8;
        if (opflow->allow_lineflow_violation) {
          entries_per_line += 2; /* 1 slack entry per row */
        }
        nnz_ineqjacsp += entries_per_line;
        linej++;
      }
    }
  }

  /* ---- Hessian nnz counting ---- */
  {
    std::map<std::pair<int, int>, int> existing_pairs;

    // Bus equality constraint Hessian (1 diagonal entry)
    for (int ibus = 0; ibus < busparams->nbus; ++ibus) {
      const int xloc = busparams->xidx[ibus];
      busparams->hesssp_eq_idx[ibus] =
          count_entry(existing_pairs, xloc + 1, xloc + 1, nnz_hesssp);
    }

    // Line equality constraints Hessian (4x4, 10 upper triangular)
    for (int iline = 0; iline < lineparams->nlineON; ++iline) {
      const int xlocf = lineparams->xidxf[iline];
      const int xloct = lineparams->xidxt[iline];
      const int base = 10 * iline;

      lineparams->hesssp_eq_idx[base + 0] =
          count_entry(existing_pairs, xlocf, xlocf, nnz_hesssp);
      lineparams->hesssp_eq_idx[base + 1] =
          count_entry(existing_pairs, xlocf, xlocf + 1, nnz_hesssp);
      lineparams->hesssp_eq_idx[base + 2] =
          count_entry(existing_pairs, xlocf, xloct, nnz_hesssp);
      lineparams->hesssp_eq_idx[base + 3] =
          count_entry(existing_pairs, xlocf, xloct + 1, nnz_hesssp);

      lineparams->hesssp_eq_idx[base + 4] =
          count_entry(existing_pairs, xlocf + 1, xlocf + 1, nnz_hesssp);
      lineparams->hesssp_eq_idx[base + 5] =
          count_entry(existing_pairs, xlocf + 1, xloct, nnz_hesssp);
      lineparams->hesssp_eq_idx[base + 6] =
          count_entry(existing_pairs, xlocf + 1, xloct + 1, nnz_hesssp);

      lineparams->hesssp_eq_idx[base + 7] =
          count_entry(existing_pairs, xloct, xloct, nnz_hesssp);
      lineparams->hesssp_eq_idx[base + 8] =
          count_entry(existing_pairs, xloct, xloct + 1, nnz_hesssp);

      lineparams->hesssp_eq_idx[base + 9] =
          count_entry(existing_pairs, xloct + 1, xloct + 1, nnz_hesssp);
    }

    // Generator AGC inequality constraints Hessian (3 upper triangular entries)
    if (opflow->has_gensetpoint && opflow->use_agc) {
      const int xloc_dpsys = pbpolrajahiopsparse->agc_xidx;

      for (int g = 0; g < genparams->ngenON; ++g) {
        if (genparams->isrenewable[g])
          continue;

        const int xloc_pg = genparams->xidx[g];
        const int xloc_dev = genparams->xpdevidx[g];

        const int base = 3 * g;

        genparams->hesssp_ineq_idx[base + 0] =
            count_entry(existing_pairs, xloc_pg, xloc_pg, nnz_hesssp);
        genparams->hesssp_ineq_idx[base + 1] =
            count_entry(existing_pairs, xloc_pg, xloc_dev, nnz_hesssp);
        genparams->hesssp_ineq_idx[base + 2] =
            count_entry(existing_pairs, xloc_pg, xloc_dpsys, nnz_hesssp);
      }
    }

    // Set voltage inequality constraints Hessian (1 entry)
    if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
      for (int ibus = 0; ibus < busparams->nbus; ++ibus) {
        if (!(busparams->ispv[ibus] || busparams->isref[ibus]))
          continue;

        const int xloc_v = busparams->xidx[ibus] + 1;
        const int goff = busparams->genoffset[ibus];
        const int ngen = busparams->ngenONbus[ibus];

        for (int k = 0; k < ngen; ++k) {
          const int g = goff + k;
          const int xloc_qg = genparams->xidx[g] + 1;

          busparams->hesssp_ineq_idx[g] =
              count_entry(existing_pairs, xloc_qg, xloc_v, nnz_hesssp);
        }
      }
    }

    // Line inequality constraints Hessian (4x4, 10 upper triangular)
    for (int imon = 0; imon < lineparams->nlinelim; ++imon) {
      const int iline = lineparams->linelimidx[imon];
      if (lineparams->isdcline[iline])
        continue;

      const int xlocf = lineparams->xidxf[iline];
      const int xloct = lineparams->xidxt[iline];
      const int base = 10 * imon;

      lineparams->hesssp_ineq_idx[base + 0] =
          count_entry(existing_pairs, xlocf, xlocf, nnz_hesssp);
      lineparams->hesssp_ineq_idx[base + 1] =
          count_entry(existing_pairs, xlocf, xlocf + 1, nnz_hesssp);
      lineparams->hesssp_ineq_idx[base + 2] =
          count_entry(existing_pairs, xlocf, xloct, nnz_hesssp);
      lineparams->hesssp_ineq_idx[base + 3] =
          count_entry(existing_pairs, xlocf, xloct + 1, nnz_hesssp);

      lineparams->hesssp_ineq_idx[base + 4] =
          count_entry(existing_pairs, xlocf + 1, xlocf + 1, nnz_hesssp);
      lineparams->hesssp_ineq_idx[base + 5] =
          count_entry(existing_pairs, xlocf + 1, xloct, nnz_hesssp);
      lineparams->hesssp_ineq_idx[base + 6] =
          count_entry(existing_pairs, xlocf + 1, xloct + 1, nnz_hesssp);

      lineparams->hesssp_ineq_idx[base + 7] =
          count_entry(existing_pairs, xloct, xloct, nnz_hesssp);
      lineparams->hesssp_ineq_idx[base + 8] =
          count_entry(existing_pairs, xloct, xloct + 1, nnz_hesssp);

      lineparams->hesssp_ineq_idx[base + 9] =
          count_entry(existing_pairs, xloct + 1, xloct + 1, nnz_hesssp);
    }

    // Power-imbalance objective Hessian (2 diagonal entries)
    if (opflow->include_powerimbalance_variables) {
      for (int ibus = 0; ibus < busparams->nbus; ++ibus) {
        const int xloc = busparams->xidxpimb[ibus];
        const int base = 2 * ibus;

        busparams->hesssp_obj_idx[base + 0] =
            count_entry(existing_pairs, xloc, xloc, nnz_hesssp);
        busparams->hesssp_obj_idx[base + 1] =
            count_entry(existing_pairs, xloc + 1, xloc + 1, nnz_hesssp);
      }
    }

    // Gen objective Hessian (1 diagonal entry)
    if (opflow->objectivetype == MIN_GEN_COST ||
        opflow->objectivetype == MIN_GENSETPOINT_DEVIATION) {
      for (int igen = 0; igen < genparams->ngenON; ++igen) {
        const int xloc = (opflow->objectivetype == MIN_GEN_COST)
                             ? genparams->xidx[igen]
                             : genparams->xpdevidx[igen];

        genparams->hesssp_obj_idx[igen] =
            count_entry(existing_pairs, xloc, xloc, nnz_hesssp);
      }
    }

    // Load objective Hessian (2 diagonal entries)
    if (opflow->include_loadloss_variables) {
      for (int iload = 0; iload < loadparams->nload; ++iload) {
        const int xloc = loadparams->xidx[iload];
        const int base = 2 * iload;

        loadparams->hesssp_obj_idx[base + 0] =
            count_entry(existing_pairs, xloc, xloc, nnz_hesssp);
        loadparams->hesssp_obj_idx[base + 1] =
            count_entry(existing_pairs, xloc + 1, xloc + 1, nnz_hesssp);
      }
    }
  }

  /* Store nnz counts */
  opflow->nnz_eqjacsp = nnz_eqjacsp;
  opflow->nnz_ineqjacsp = nnz_ineqjacsp;
  opflow->nnz_hesssp = nnz_hesssp;

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
      OPFLOWComputeEqualityConstraintJacobian_PBPOL;
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
