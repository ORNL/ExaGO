
#include <exago_config.h>

#if defined(EXAGO_ENABLE_RAJA)
#if defined(EXAGO_ENABLE_HIOP_SPARSE)

#include <RAJA/RAJA.hpp>
#include <private/raja_exec_config.h>

#include "pbpolrajahiopsparse_gpu.hpp"

void ComputeIneqJacValuesGPU_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                                 const double *x_dev,
                                                 double *jacd_dev) {
  PbpolModelRajaHiop *pbpolrajahiopsparse =
      reinterpret_cast<PbpolModelRajaHiop *>(opflow->model);
  GENParamsRajaHiop *genparams = &pbpolrajahiopsparse->genparams;
  BUSParamsRajaHiop *busparams = &pbpolrajahiopsparse->busparams;

  /*
   * Generator set-point inequality Jacobian (has_gensetpoint && use_agc).
   *
   * PETSc reference (pbpol.cpp:1191-1232):
   *   Row gloc:   [apf*delP - delPg,  -(Pg - pt),  apf*(Pg - pt)]
   *   Row gloc+1: [apf*delP - delPg,  (pb - Pg),  -apf*(pb - Pg)]
   *
   * Flat array layout per generator (6 entries):
   *   [0..2] = row 0 values (Pg, delPg, delP columns)
   *   [3..5] = row 1 values (Pg, delPg, delP columns)
   */
  if (opflow->has_gensetpoint && opflow->use_agc) {
    int *g_ineqjacsp_idx = genparams->ineqjacspgen_idx_dev_;
    int *g_xidx = genparams->xidx_dev_;
    int *g_xpdevidx = genparams->xpdevidx_dev_;
    int *g_isrenewable = genparams->isrenewable_dev_;
    double *g_apf = genparams->apf_dev_;
    double *g_pt = genparams->pt_dev_;
    double *g_pb = genparams->pb_dev_;
    int agc_xidx = pbpolrajahiopsparse->agc_xidx;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, genparams->ngenON),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if (g_isrenewable[i])
            return;

          int base = g_ineqjacsp_idx[i];
          double Pg = x_dev[g_xidx[i]];
          double delPg = x_dev[g_xpdevidx[i]];
          double delP = x_dev[agc_xidx];
          double apf = g_apf[i];
          double pt = g_pt[i];
          double pb = g_pb[i];

          double apf_delP_minus_delPg = apf * delP - delPg;

          /* Row 0: d/d{Pg, delPg, delP} of (apf*delP - delPg)*(Pg - pt) */
          jacd_dev[base + 0] = apf_delP_minus_delPg;
          jacd_dev[base + 1] = -(Pg - pt);
          jacd_dev[base + 2] = apf * (Pg - pt);

          /* Row 1: d/d{Pg, delPg, delP} of (delPg - apf*delP)*(pb - Pg) */
          jacd_dev[base + 3] = apf_delP_minus_delPg;
          jacd_dev[base + 4] = pb - Pg;
          jacd_dev[base + 5] = -apf * (pb - Pg);
        });
  }

  /*
   * FIXED_WITHIN_QBOUNDS voltage constraint Jacobian.
   *
   * PETSc reference (pbpol.cpp:1235-1278):
   *   For each PV/REF bus with ngenON generators:
   *     Row 0 (gloc):   [Vset_0-V, Vset_1-V, ..., Qmax-Q]
   *     Row 1 (gloc+1): [Vset_0-V, Vset_1-V, ..., Qmin-Q]
   *
   * Flat array layout per bus (2 * (ngenON + 1) entries):
   *   Row 0: ngenON Qg derivative entries + 1 V derivative entry
   *   Row 1: ngenON Qg derivative entries + 1 V derivative entry
   *
   * This kernel parallelizes over buses. The inner loop over generators
   * is sequential within each thread (variable-length per bus).
   */
  if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
    int *b_ispv = busparams->ispv_dev_;
    int *b_isref = busparams->isref_dev_;
    int *b_xidx = busparams->xidx_dev_;
    int *b_ineqjacsp_idx = busparams->ineqjacsp_idx_dev_;
    int *b_genoffset = busparams->genoffset_dev_;
    int *b_ngenONbus = busparams->ngenONbus_dev_;
    int *g_xidx = genparams->xidx_dev_;
    double *g_qt = genparams->qt_dev_;
    double *g_qb = genparams->qb_dev_;
    double *g_vs = genparams->vs_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if (!b_ispv[i] && !b_isref[i])
            return;

          int base = b_ineqjacsp_idx[i];
          int ngen = b_ngenONbus[i];
          int goff = b_genoffset[i];
          double V = x_dev[b_xidx[i] + 1];

          double Q = 0.0, Qmax = 0.0, Qmin = 0.0;
          double Vset = 0.0;

          /* Row 0 and Row 1: Qg derivative entries */
          for (int k = 0; k < ngen; k++) {
            int gidx = goff + k;
            double Qg = x_dev[g_xidx[gidx] + 1];
            Q += Qg;
            Qmax += g_qt[gidx];
            Qmin += g_qb[gidx];
            Vset = g_vs[gidx];

            double dQg = Vset - V;
            /* Row 0 entry for this gen's Qg */
            jacd_dev[base + k] = dQg;
            /* Row 1 entry for this gen's Qg */
            jacd_dev[base + ngen + 1 + k] = dQg;
          }

          /* Row 0: V derivative entry (last in this row) */
          jacd_dev[base + ngen] = Qmax - Q;
          /* Row 1: V derivative entry (last in this row) */
          jacd_dev[base + ngen + 1 + ngen] = Qmin - Q;
        });
  }

  /*
   * Line flow constraint Jacobian.
   *
   * Computes dSf2/d{thetaf,Vmf,thetat,Vmt} and dSt2/d{thetaf,Vmf,thetat,Vmt}
   * for each monitored line, plus optional slack variable entries (-1.0).
   *
   * Flat array layout per line (8 entries without slack, 10 with):
   *   Row 0 (Sf2): [dSf2_dthetaf, dSf2_dVmf, dSf2_dthetat, dSf2_dVmt,
   *                  (-1.0 if slack)]
   *   Row 1 (St2): [dSt2_dthetaf, dSt2_dVmf, dSt2_dthetat, dSt2_dVmt,
   *                  (-1.0 if slack)]
   */
  LINEParamsRajaHiop *lineparams = &pbpolrajahiopsparse->lineparams;

  if (lineparams->nlinelim) {
    double *Gff_arr = lineparams->Gff_dev_;
    double *Bff_arr = lineparams->Bff_dev_;
    double *Gft_arr = lineparams->Gft_dev_;
    double *Bft_arr = lineparams->Bft_dev_;
    double *Gtf_arr = lineparams->Gtf_dev_;
    double *Btf_arr = lineparams->Btf_dev_;
    double *Gtt_arr = lineparams->Gtt_dev_;
    double *Btt_arr = lineparams->Btt_dev_;
    int *linelimidx = lineparams->linelimidx_dev_;
    int *xidxf = lineparams->xidxf_dev_;
    int *xidxt = lineparams->xidxt_dev_;
    int *ineqjacsp_idx = lineparams->ineqjacsp_idx_dev_;
    int has_slack = (int)opflow->allow_lineflow_violation;
    int row_stride = 4 + has_slack;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, lineparams->nlinelim),
        RAJA_LAMBDA(RAJA::Index_type i) {
          int j = linelimidx[i];
          int base = ineqjacsp_idx[i];

          double thetaf = x_dev[xidxf[j]];
          double Vmf = x_dev[xidxf[j] + 1];
          double thetat = x_dev[xidxt[j]];
          double Vmt = x_dev[xidxt[j] + 1];
          double thetaft = thetaf - thetat;
          double thetatf = thetat - thetaf;

          double Gff = Gff_arr[j], Bff = Bff_arr[j];
          double Gft = Gft_arr[j], Bft = Bft_arr[j];
          double Gtf = Gtf_arr[j], Btf = Btf_arr[j];
          double Gtt = Gtt_arr[j], Btt = Btt_arr[j];

          double sin_ft = sin(thetaft), cos_ft = cos(thetaft);
          double sin_tf = sin(thetatf), cos_tf = cos(thetatf);

          double Pf =
              Gff * Vmf * Vmf + Vmf * Vmt * (Gft * cos_ft + Bft * sin_ft);
          double Qf =
              -Bff * Vmf * Vmf + Vmf * Vmt * (-Bft * cos_ft + Gft * sin_ft);
          double Pt =
              Gtt * Vmt * Vmt + Vmt * Vmf * (Gtf * cos_tf + Btf * sin_tf);
          double Qt =
              -Btt * Vmt * Vmt + Vmt * Vmf * (-Btf * cos_tf + Gtf * sin_tf);

          double dSf2_dPf = 2 * Pf, dSf2_dQf = 2 * Qf;
          double dSt2_dPt = 2 * Pt, dSt2_dQt = 2 * Qt;

          double dPf_dthetaf = Vmf * Vmt * (-Gft * sin_ft + Bft * cos_ft);
          double dPf_dVmf = 2 * Gff * Vmf + Vmt * (Gft * cos_ft + Bft * sin_ft);
          double dPf_dthetat = Vmf * Vmt * (Gft * sin_ft - Bft * cos_ft);
          double dPf_dVmt = Vmf * (Gft * cos_ft + Bft * sin_ft);

          double dQf_dthetaf = Vmf * Vmt * (Bft * sin_ft + Gft * cos_ft);
          double dQf_dVmf =
              -2 * Bff * Vmf + Vmt * (-Bft * cos_ft + Gft * sin_ft);
          double dQf_dthetat = Vmf * Vmt * (-Bft * sin_ft - Gft * cos_ft);
          double dQf_dVmt = Vmf * (-Bft * cos_ft + Gft * sin_ft);

          double dPt_dthetat = Vmt * Vmf * (-Gtf * sin_tf + Btf * cos_tf);
          double dPt_dVmt = 2 * Gtt * Vmt + Vmf * (Gtf * cos_tf + Btf * sin_tf);
          double dPt_dthetaf = Vmt * Vmf * (Gtf * sin_tf - Btf * cos_tf);
          double dPt_dVmf = Vmt * (Gtf * cos_tf + Btf * sin_tf);

          double dQt_dthetat = Vmt * Vmf * (Btf * sin_tf + Gtf * cos_tf);
          double dQt_dVmt =
              -2 * Btt * Vmt + Vmf * (-Btf * cos_tf + Gtf * sin_tf);
          double dQt_dthetaf = Vmt * Vmf * (-Btf * sin_tf - Gtf * cos_tf);
          double dQt_dVmf = Vmt * (-Btf * cos_tf + Gtf * sin_tf);

          /* Row 0 (Sf2): derivatives w.r.t. thetaf, Vmf, thetat, Vmt */
          jacd_dev[base + 0] = dSf2_dPf * dPf_dthetaf + dSf2_dQf * dQf_dthetaf;
          jacd_dev[base + 1] = dSf2_dPf * dPf_dVmf + dSf2_dQf * dQf_dVmf;
          jacd_dev[base + 2] = dSf2_dPf * dPf_dthetat + dSf2_dQf * dQf_dthetat;
          jacd_dev[base + 3] = dSf2_dPf * dPf_dVmt + dSf2_dQf * dQf_dVmt;

          /* Row 1 (St2): derivatives w.r.t. thetaf, Vmf, thetat, Vmt */
          jacd_dev[base + row_stride + 0] =
              dSt2_dPt * dPt_dthetaf + dSt2_dQt * dQt_dthetaf;
          jacd_dev[base + row_stride + 1] =
              dSt2_dPt * dPt_dVmf + dSt2_dQt * dQt_dVmf;
          jacd_dev[base + row_stride + 2] =
              dSt2_dPt * dPt_dthetat + dSt2_dQt * dQt_dthetat;
          jacd_dev[base + row_stride + 3] =
              dSt2_dPt * dPt_dVmt + dSt2_dQt * dQt_dVmt;

          /* Slack variable entries (Step 5) */
          if (has_slack) {
            jacd_dev[base + 4] = -1.0;
            jacd_dev[base + row_stride + 4] = -1.0;
          }
        });
  }
}

void ComputeEqJacValuesGPU_PBPOLRAJAHIOPSPARSE(OPFLOW opflow,
                                               const double *x_dev,
                                               double *jace_dev) {
  PbpolModelRajaHiop *pbpolrajahiopsparse =
      reinterpret_cast<PbpolModelRajaHiop *>(opflow->model);
  BUSParamsRajaHiop *busparams = &pbpolrajahiopsparse->busparams;
  GENParamsRajaHiop *genparams = &pbpolrajahiopsparse->genparams;
  LOADParamsRajaHiop *loadparams = &pbpolrajahiopsparse->loadparams;
  LINEParamsRajaHiop *lineparams = &pbpolrajahiopsparse->lineparams;

  /* Zero the equality Jacobian values before accumulating */
  RAJA::forall<exago_raja_exec>(
      RAJA::RangeSegment(0, opflow->nnz_eqjacsp),
      RAJA_LAMBDA(RAJA::Index_type i) { jace_dev[i] = 0.0; });

  /*
   * Kernel 1: Bus contributions (shunt self-admittance, power imbalance,
   *           generator Pg/Qg, load loss, gen setpoint).
   *
   * For bus self-admittance entries, the shunt values are written directly.
   * Line diagonal contributions are accumulated via atomicAdd by Kernel 2.
   */
  {
    int *b_isisolated = busparams->isisolated_dev_;
    int *b_xidx = busparams->xidx_dev_;
    double *b_gl = busparams->gl_dev_;
    double *b_bl = busparams->bl_dev_;
    int *b_selfidx = busparams->eqjacsp_selfidx_dev_;
    /*KS: can probably parallelize this differently but it is correct  --
     * optimizing existing code is easier than writing it froms scratch .... */
    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type i) {
          int pbase = b_selfidx[2 * i];
          int qbase = b_selfidx[2 * i + 1];

          if (b_isisolated[i]) {
            jace_dev[pbase + 0] = 1.0;
            jace_dev[pbase + 1] = 0.0;
            jace_dev[qbase + 0] = 0.0;
            jace_dev[qbase + 1] = 1.0;
            return;
          }

          double Vm = x_dev[b_xidx[i] + 1];

          jace_dev[pbase + 0] = 0.0;
          jace_dev[pbase + 1] = 2.0 * Vm * b_gl[i];
          jace_dev[qbase + 0] = 0.0;
          jace_dev[qbase + 1] = -2.0 * Vm * b_bl[i];
        });

    /* Power imbalance: constant entries */
    if (opflow->include_powerimbalance_variables) {
      int *b_jacsp = busparams->jacsp_idx_dev_;
      int *b_jacsq = busparams->jacsq_idx_dev_;
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, busparams->nbus),
          RAJA_LAMBDA(RAJA::Index_type i) {
            jace_dev[b_jacsp[i]] = 1.0;
            jace_dev[b_jacsp[i] + 1] = -1.0;
            jace_dev[b_jacsq[i]] = 1.0;
            jace_dev[b_jacsq[i] + 1] = -1.0;
          });
    }

    /* Generator Pg/Qg contributions: -1 per active generator */
    int *g_eqjacspbus = genparams->eqjacspbus_idx_dev_;
    int *g_eqjacsqbus = genparams->eqjacsqbus_idx_dev_;
    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, genparams->ngenON),
        RAJA_LAMBDA(RAJA::Index_type i) {
          jace_dev[g_eqjacspbus[i]] = -1.0;
          jace_dev[g_eqjacsqbus[i]] = -1.0;
        });

    /* Load loss contributions: -1 per load */
    if (opflow->include_loadloss_variables) {
      int *l_jacsp = loadparams->jacsp_idx_dev_;
      int *l_jacsq = loadparams->jacsq_idx_dev_;
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, loadparams->nload),
          RAJA_LAMBDA(RAJA::Index_type i) {
            jace_dev[l_jacsp[i]] = -1.0;
            jace_dev[l_jacsq[i]] = -1.0;
          });
    }

    /* Generator set-point equality constraints: constant entries */
    if (opflow->has_gensetpoint) {
      int *g_eqjacspgen = genparams->eqjacspgen_idx_dev_;
      int *g_isrenewable = genparams->isrenewable_dev_;
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, genparams->ngenON),
          RAJA_LAMBDA(RAJA::Index_type i) {
            if (g_isrenewable[i])
              return;
            int base = g_eqjacspgen[i];
            jace_dev[base + 0] = -1.0;
            jace_dev[base + 1] = 1.0;
            jace_dev[base + 2] = 1.0;
            jace_dev[base + 3] = 1.0;
          });
    }
  }

  /*
   * Kernel 2: Line contributions.
   *
   * Off-diagonal entries (remote-bus columns) are written directly.
   * Diagonal entries (self-bus columns) are accumulated with atomicAdd
   * into the bus self-admittance positions set by Kernel 1.
   */
  {
    double *l_Gff = lineparams->Gff_dev_;
    double *l_Bff = lineparams->Bff_dev_;
    double *l_Gft = lineparams->Gft_dev_;
    double *l_Bft = lineparams->Bft_dev_;
    double *l_Gtf = lineparams->Gtf_dev_;
    double *l_Btf = lineparams->Btf_dev_;
    double *l_Gtt = lineparams->Gtt_dev_;
    double *l_Btt = lineparams->Btt_dev_;
    int *l_xidxf = lineparams->xidxf_dev_;
    int *l_xidxt = lineparams->xidxt_dev_;
    int *l_eqjacsp_idx = lineparams->eqjacsp_idx_dev_;
    int *l_eqjacsp_diag = lineparams->eqjacsp_diag_idx_dev_;
    int *l_isdcline = lineparams->isdcline_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, lineparams->nlineON),
        RAJA_LAMBDA(RAJA::Index_type l) {
          if (l_isdcline[l])
            return;

          double thetaf = x_dev[l_xidxf[l]];
          double Vmf = x_dev[l_xidxf[l] + 1];
          double thetat = x_dev[l_xidxt[l]];
          double Vmt = x_dev[l_xidxt[l] + 1];
          double thetaft = thetaf - thetat;
          double thetatf = thetat - thetaf;

          double Gff = l_Gff[l], Bff = l_Bff[l];
          double Gft = l_Gft[l], Bft = l_Bft[l];
          double Gtf = l_Gtf[l], Btf = l_Btf[l];
          double Gtt = l_Gtt[l], Btt = l_Btt[l];

          double sin_ft = sin(thetaft), cos_ft = cos(thetaft);
          double sin_tf = sin(thetatf), cos_tf = cos(thetatf);

          /* From-bus diagonal derivatives (accumulate into bus self entries) */
          double dPf_dthetaf = Vmf * Vmt * (-Gft * sin_ft + Bft * cos_ft);
          double dPf_dVmf =
              2.0 * Gff * Vmf + Vmt * (Gft * cos_ft + Bft * sin_ft);
          double dQf_dthetaf = Vmf * Vmt * (Bft * sin_ft + Gft * cos_ft);
          double dQf_dVmf =
              -2.0 * Bff * Vmf + Vmt * (-Bft * cos_ft + Gft * sin_ft);

          int pfbase = l_eqjacsp_diag[4 * l + 0];
          int qfbase = l_eqjacsp_diag[4 * l + 1];

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[pfbase + 0],
                                             dPf_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[pfbase + 1], dPf_dVmf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qfbase + 0],
                                             dQf_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qfbase + 1], dQf_dVmf);

          /* From-bus off-diagonal derivatives (atomicAdd for parallel lines) */
          double dPf_dthetat = Vmf * Vmt * (Gft * sin_ft - Bft * cos_ft);
          double dPf_dVmt = Vmf * (Gft * cos_ft + Bft * sin_ft);
          double dQf_dthetat = Vmf * Vmt * (-Bft * sin_ft - Gft * cos_ft);
          double dQf_dVmt = Vmf * (-Bft * cos_ft + Gft * sin_ft);

          int obase = l_eqjacsp_idx[l];
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 0], dPf_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 1], dPf_dVmt);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 2], dQf_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 3], dQf_dVmt);

          /* To-bus diagonal derivatives */
          double dPt_dthetat = Vmt * Vmf * (-Gtf * sin_tf + Btf * cos_tf);
          double dPt_dVmt =
              2.0 * Gtt * Vmt + Vmf * (Gtf * cos_tf + Btf * sin_tf);
          double dQt_dthetat = Vmt * Vmf * (Btf * sin_tf + Gtf * cos_tf);
          double dQt_dVmt =
              -2.0 * Btt * Vmt + Vmf * (-Btf * cos_tf + Gtf * sin_tf);

          int ptbase = l_eqjacsp_diag[4 * l + 2];
          int qtbase = l_eqjacsp_diag[4 * l + 3];

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[ptbase + 0],
                                             dPt_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[ptbase + 1], dPt_dVmt);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qtbase + 0],
                                             dQt_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qtbase + 1], dQt_dVmt);

          /* To-bus off-diagonal derivatives (atomicAdd for parallel lines) */
          double dPt_dthetaf = Vmt * Vmf * (Gtf * sin_tf - Btf * cos_tf);
          double dPt_dVmf = Vmt * (Gtf * cos_tf + Btf * sin_tf);
          double dQt_dthetaf = Vmt * Vmf * (-Btf * sin_tf - Gtf * cos_tf);
          double dQt_dVmf = Vmt * (-Btf * cos_tf + Gtf * sin_tf);

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 4], dPt_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 5], dPt_dVmf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 6], dQt_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 7], dQt_dVmf);
        });
  }
}

#endif // EXAGO_ENABLE_HIOP_SPARSE
#endif // EXAGO_ENABLE_RAJA
