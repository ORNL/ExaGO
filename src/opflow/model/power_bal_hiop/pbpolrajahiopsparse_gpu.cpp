
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
                                               const int *perm_dev,
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
   */
  {
    int *b_isisolated = busparams->isisolated_dev_;
    int *b_xidx = busparams->xidx_dev_;
    double *b_gl = busparams->gl_dev_;
    double *b_bl = busparams->bl_dev_;
    int *b_idx = busparams->eqjacsp_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type i) {
          int pbase = b_idx[2 * i];
          int qbase = b_idx[2 * i + 1];

          if (b_isisolated[i]) {
            jace_dev[perm_dev[pbase + 0]] = 1.0;
            jace_dev[perm_dev[pbase + 1]] = 0.0;
            jace_dev[perm_dev[qbase + 0]] = 0.0;
            jace_dev[perm_dev[qbase + 1]] = 1.0;
            return;
          }

          double Vm = x_dev[b_xidx[i] + 1];

          jace_dev[perm_dev[pbase + 0]] = 0.0;
          jace_dev[perm_dev[pbase + 1]] = 2.0 * Vm * b_gl[i];
          jace_dev[perm_dev[qbase + 0]] = 0.0;
          jace_dev[perm_dev[qbase + 1]] = -2.0 * Vm * b_bl[i];
        });

    if (opflow->include_powerimbalance_variables) {
      int *b_jacsp = busparams->jacsp_idx_dev_;
      int *b_jacsq = busparams->jacsq_idx_dev_;
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, busparams->nbus),
          RAJA_LAMBDA(RAJA::Index_type i) {
            jace_dev[perm_dev[b_jacsp[i]]] = 1.0;
            jace_dev[perm_dev[b_jacsp[i]] + 1] = -1.0;
            jace_dev[perm_dev[b_jacsq[i]]] = 1.0;
            jace_dev[perm_dev[b_jacsq[i]] + 1] = -1.0;
          });
    }

    int *g_eqjacspbus = genparams->eqjacspbus_idx_dev_;
    int *g_eqjacsqbus = genparams->eqjacsqbus_idx_dev_;
    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, genparams->ngenON),
        RAJA_LAMBDA(RAJA::Index_type i) {
          jace_dev[perm_dev[g_eqjacspbus[i]]] = -1.0;
          jace_dev[perm_dev[g_eqjacsqbus[i]]] = -1.0;
        });

    if (opflow->include_loadloss_variables) {
      int *l_jacsp = loadparams->jacsp_idx_dev_;
      int *l_jacsq = loadparams->jacsq_idx_dev_;
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, loadparams->nload),
          RAJA_LAMBDA(RAJA::Index_type i) {
            jace_dev[perm_dev[l_jacsp[i]]] = -1.0;
            jace_dev[perm_dev[l_jacsq[i]]] = -1.0;
          });
    }

    if (opflow->has_gensetpoint) {
      int *g_eqjacspgen = genparams->eqjacspgen_idx_dev_;
      int *g_isrenewable = genparams->isrenewable_dev_;
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, genparams->ngenON),
          RAJA_LAMBDA(RAJA::Index_type i) {
            if (g_isrenewable[i])
              return;
            int base = g_eqjacspgen[i];
            jace_dev[perm_dev[base + 0]] = -1.0;
            jace_dev[perm_dev[base + 1]] = 1.0;
            jace_dev[perm_dev[base + 2]] = 1.0;
            jace_dev[perm_dev[base + 3]] = 1.0;
          });
    }
  }

  /*
   * Kernel 2: Line contributions — diagonal via atomicAdd, off-diagonal
   * via atomicAdd (for parallel lines sharing positions).
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

          /* From-bus diagonal */
          double dPf_dthetaf = Vmf * Vmt * (-Gft * sin_ft + Bft * cos_ft);
          double dPf_dVmf =
              2.0 * Gff * Vmf + Vmt * (Gft * cos_ft + Bft * sin_ft);
          double dQf_dthetaf = Vmf * Vmt * (Bft * sin_ft + Gft * cos_ft);
          double dQf_dVmf =
              -2.0 * Bff * Vmf + Vmt * (-Bft * cos_ft + Gft * sin_ft);

          int pfbase = l_eqjacsp_diag[4 * l + 0];
          int qfbase = l_eqjacsp_diag[4 * l + 1];

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[pfbase + 0]],
                                             dPf_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[pfbase + 1]],
                                             dPf_dVmf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[qfbase + 0]],
                                             dQf_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[qfbase + 1]],
                                             dQf_dVmf);

          /* From-bus off-diagonal */
          double dPf_dthetat = Vmf * Vmt * (Gft * sin_ft - Bft * cos_ft);
          double dPf_dVmt = Vmf * (Gft * cos_ft + Bft * sin_ft);
          double dQf_dthetat = Vmf * Vmt * (-Bft * sin_ft - Gft * cos_ft);
          double dQf_dVmt = Vmf * (-Bft * cos_ft + Gft * sin_ft);

          int obase = l_eqjacsp_idx[l];
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 0]],
                                             dPf_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 1]],
                                             dPf_dVmt);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 2]],
                                             dQf_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 3]],
                                             dQf_dVmt);

          /* To-bus diagonal */
          double dPt_dthetat = Vmt * Vmf * (-Gtf * sin_tf + Btf * cos_tf);
          double dPt_dVmt =
              2.0 * Gtt * Vmt + Vmf * (Gtf * cos_tf + Btf * sin_tf);
          double dQt_dthetat = Vmt * Vmf * (Btf * sin_tf + Gtf * cos_tf);
          double dQt_dVmt =
              -2.0 * Btt * Vmt + Vmf * (-Btf * cos_tf + Gtf * sin_tf);

          int ptbase = l_eqjacsp_diag[4 * l + 2];
          int qtbase = l_eqjacsp_diag[4 * l + 3];

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[ptbase + 0]],
                                             dPt_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[ptbase + 1]],
                                             dPt_dVmt);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[qtbase + 0]],
                                             dQt_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[qtbase + 1]],
                                             dQt_dVmt);

          /* To-bus off-diagonal */
          double dPt_dthetaf = Vmt * Vmf * (Gtf * sin_tf - Btf * cos_tf);
          double dPt_dVmf = Vmt * (Gtf * cos_tf + Btf * sin_tf);
          double dQt_dthetaf = Vmt * Vmf * (-Btf * sin_tf - Gtf * cos_tf);
          double dQt_dVmf = Vmt * (-Btf * cos_tf + Gtf * sin_tf);

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 4]],
                                             dPt_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 5]],
                                             dPt_dVmf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 6]],
                                             dQt_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[perm_dev[obase + 7]],
                                             dQt_dVmf);
        });
  }
}

void ComputeHessValuesGPU_PBPOLRAJAHIOPSPARSE(
    OPFLOW opflow, const double *x_dev, const double *lambdae_dev,
    const double *lambdai_dev, const int *perm_dev, double *hess_dev) {

  PbpolModelRajaHiop *pbpolrajahiopsparse =
      reinterpret_cast<PbpolModelRajaHiop *>(opflow->model);

  PS ps = opflow->ps;
  BUSParamsRajaHiop *busparams = &pbpolrajahiopsparse->busparams;
  GENParamsRajaHiop *genparams = &pbpolrajahiopsparse->genparams;
  LOADParamsRajaHiop *loadparams = &pbpolrajahiopsparse->loadparams;
  LINEParamsRajaHiop *lineparams = &pbpolrajahiopsparse->lineparams;

  /* Zero the Hessian values before accumulating */
  RAJA::forall<exago_raja_exec>(
      RAJA::RangeSegment(0, opflow->nnz_hesssp),
      RAJA_LAMBDA(RAJA::Index_type i) { hess_dev[i] = 0.0; });

  // Bus equality constraint Hessian (1 diagonal entry)
  {
    int *bus_gidx = busparams->gidx_dev_;
    double *bus_gl = busparams->gl_dev_;
    double *bus_bl = busparams->bl_dev_;
    int *bus_hesssp_eq_idx = busparams->hesssp_eq_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type ibus) {
          int gloc = bus_gidx[ibus];

          double lambdae_gloc = lambdae_dev[gloc];
          double lambdae_gloc1 = lambdae_dev[gloc + 1];

          double val = lambdae_gloc * (2.0 * bus_gl[ibus]) +
                       lambdae_gloc1 * (-2.0 * bus_bl[ibus]);

          hess_dev[perm_dev[bus_hesssp_eq_idx[ibus]]] += val;
        });
  }

  // Line equality constraints Hessian (4x4, 10 upper triangular)
  {
    int *line_xidxf = lineparams->xidxf_dev_;
    int *line_xidxt = lineparams->xidxt_dev_;
    int *line_geqidxf = lineparams->geqidxf_dev_;
    int *line_geqidxt = lineparams->geqidxt_dev_;
    int *line_hesssp_eq_idx = lineparams->hesssp_eq_idx_dev_;

    double *line_Gff = lineparams->Gff_dev_;
    double *line_Bff = lineparams->Bff_dev_;
    double *line_Gft = lineparams->Gft_dev_;
    double *line_Bft = lineparams->Bft_dev_;
    double *line_Gtf = lineparams->Gtf_dev_;
    double *line_Btf = lineparams->Btf_dev_;
    double *line_Gtt = lineparams->Gtt_dev_;
    double *line_Btt = lineparams->Btt_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, lineparams->nlineON),
        RAJA_LAMBDA(RAJA::Index_type iline) {
          int xlocf = line_xidxf[iline];
          int xloct = line_xidxt[iline];
          int base = 10 * iline;

          int glocf = line_geqidxf[iline];
          int gloct = line_geqidxt[iline];

          double lambdaPf = lambdae_dev[glocf];
          double lambdaQf = lambdae_dev[glocf + 1];
          double lambdaPt = lambdae_dev[gloct];
          double lambdaQt = lambdae_dev[gloct + 1];

          double thetaf = x_dev[xlocf];
          double Vmf = x_dev[xlocf + 1];
          double thetat = x_dev[xloct];
          double Vmt = x_dev[xloct + 1];

          double thetaft = thetaf - thetat;
          double thetatf = thetat - thetaf;

          double sin_thetaft = sin(thetaft);
          double cos_thetaft = cos(thetaft);
          double sin_thetatf = sin(thetatf);
          double cos_thetatf = cos(thetatf);

          double Gff = line_Gff[iline];
          double Bff = line_Bff[iline];
          double Gft = line_Gft[iline];
          double Bft = line_Bft[iline];
          double Gtf = line_Gtf[iline];
          double Btf = line_Btf[iline];
          double Gtt = line_Gtt[iline];
          double Btt = line_Btt[iline];

          /* dPf_dthetaf = Vmf*Vmt*(-Gft*sin_thetaft + Bft*cos_thetaft); */
          double d2Pf_00 = -Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          double d2Pf_01 = Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          double d2Pf_02 = Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          double d2Pf_03 = Vmf * (-Gft * sin_thetaft + Bft * cos_thetaft);

          /* dPf_Vmf  = 2*Gff*Vmf + Vmt*(Gft*cos_thetaft + Bft*sin_thetaft); */
          double d2Pf_11 = 2.0 * Gff;
          double d2Pf_12 = Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          double d2Pf_13 = (Gft * cos_thetaft + Bft * sin_thetaft);

          /* dPf_dthetat = Vmf*Vmt*(Gft*sin_thetaft - Bft*cos_thetaft); */
          double d2Pf_22 = Vmf * Vmt * (-Gft * cos_thetaft - Bft * sin_thetaft);
          double d2Pf_23 = Vmf * (Gft * sin_thetaft - Bft * cos_thetaft);

          /* dPf_dVmt = Vmf*(Gft*cos_thetaft + Bft*sin_thetaft); */
          double d2Pf_33 = 0.0;

          /* dQf_dthetaf = Vmf*Vmt*(Bft*sin_thetaft + Gft*cos_thetaft); */
          double d2Qf_00 = Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          double d2Qf_01 = Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          double d2Qf_02 = Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          double d2Qf_03 = Vmf * (Bft * sin_thetaft + Gft * cos_thetaft);

          /* dQf_dVmf = -2*Bff*Vmf + Vmt*(-Bft*cos_thetaft + Gft*sin_thetaft);
           */
          double d2Qf_11 = -2.0 * Bff;
          double d2Qf_12 = Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          double d2Qf_13 = (-Bft * cos_thetaft + Gft * sin_thetaft);

          /* dQf_dthetat = Vmf*Vmt*(-Bft*sin_thetaft - Gft*cos_thetaft); */
          double d2Qf_22 = Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          double d2Qf_23 = Vmf * (-Bft * sin_thetaft - Gft * cos_thetaft);

          /* dQf_dVmt = Vmf*(-Bft*cos_thetaft + Gft*sin_thetaft); */
          double d2Qf_33 = 0.0;

          /* dPt_dthetat = Vmf*Vmt*(-Gtf*sin_thetatf + Btf*cos_thetatf); */
          double d2Pt_00 = Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          double d2Pt_01 = Vmt * (Gtf * sin_thetatf - Btf * cos_thetatf);
          double d2Pt_02 = Vmf * Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);
          double d2Pt_03 = Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);

          /* dPt_Vmt  = 2*Gtt*Vmt + Vmf*(Gtf*cos_thetatf + Btf*sin_thetatf); */
          double d2Pt_11 = 0.0;
          double d2Pt_12 = Vmt * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          double d2Pt_13 = (Gtf * cos_thetatf + Btf * sin_thetatf);

          /* dPt_dthetaf = Vmf*Vmt*(Gtf*sin_thetatf - Btf*cos_thetatf); */
          double d2Pt_22 = Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          double d2Pt_23 = Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);

          /* dPt_dVmf = Vmt*(Gtf*cos_thetatf + Btf*sin_thetatf); */
          double d2Pt_33 = 2.0 * Gtt;

          /* dQt_dthetaf = Vmf*Vmt*(-Btf*sin_thetatf - Gtf*cos_thetatf); */
          double d2Qt_00 = Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          double d2Qt_01 = Vmt * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          double d2Qt_02 = Vmf * Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          double d2Qt_03 = Vmf * (-Btf * sin_thetatf - Gtf * cos_thetatf);

          /* dQt_dVmf = Vmt*(-Btf*cos_thetatf + Gtf*sin_thetatf); */
          double d2Qt_11 = 0.0;
          double d2Qt_12 = Vmt * (Btf * sin_thetatf + Gtf * cos_thetatf);
          double d2Qt_13 = (-Btf * cos_thetatf + Gtf * sin_thetatf);

          /* dQt_dthetat = Vmf*Vmt*(Btf*sin_thetatf + Gtf*cos_thetatf); */
          double d2Qt_22 = Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          double d2Qt_23 = Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);

          /* dQt_dVmt = -2*Btt*Vmt + Vmf*(-Btf*cos_thetatf + Gtf*sin_thetatf);
           */
          double d2Qt_33 = -2.0 * Btt;

          double H00 = lambdaPf * d2Pf_00 + lambdaQf * d2Qf_00 +
                       lambdaPt * d2Pt_00 + lambdaQt * d2Qt_00;
          double H01 = lambdaPf * d2Pf_01 + lambdaQf * d2Qf_01 +
                       lambdaPt * d2Pt_01 + lambdaQt * d2Qt_01;
          double H02 = lambdaPf * d2Pf_02 + lambdaQf * d2Qf_02 +
                       lambdaPt * d2Pt_02 + lambdaQt * d2Qt_02;
          double H03 = lambdaPf * d2Pf_03 + lambdaQf * d2Qf_03 +
                       lambdaPt * d2Pt_03 + lambdaQt * d2Qt_03;

          double H11 = lambdaPf * d2Pf_11 + lambdaQf * d2Qf_11 +
                       lambdaPt * d2Pt_11 + lambdaQt * d2Qt_11;
          double H12 = lambdaPf * d2Pf_12 + lambdaQf * d2Qf_12 +
                       lambdaPt * d2Pt_12 + lambdaQt * d2Qt_12;
          double H13 = lambdaPf * d2Pf_13 + lambdaQf * d2Qf_13 +
                       lambdaPt * d2Pt_13 + lambdaQt * d2Qt_13;

          double H22 = lambdaPf * d2Pf_22 + lambdaQf * d2Qf_22 +
                       lambdaPt * d2Pt_22 + lambdaQt * d2Qt_22;
          double H23 = lambdaPf * d2Pf_23 + lambdaQf * d2Qf_23 +
                       lambdaPt * d2Pt_23 + lambdaQt * d2Qt_23;

          double H33 = lambdaPf * d2Pf_33 + lambdaQf * d2Qf_33 +
                       lambdaPt * d2Pt_33 + lambdaQt * d2Qt_33;

          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 0]]], H00);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 1]]], H01);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 2]]], H02);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 3]]], H03);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 4]]], H11);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 5]]], H12);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 6]]], H13);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 7]]], H22);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 8]]], H23);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_eq_idx[base + 9]]], H33);
        });
  }

  // Generator AGC inequality constraints Hessian (3 upper triangular entries)
  if (opflow->has_gensetpoint && opflow->use_agc) {
    int *gen_isrenewable = genparams->isrenewable_dev_;
    int *gen_gineqidx = genparams->gineqidxgen_dev_;
    double *gen_apf = genparams->apf_dev_;
    int *gen_hesssp_ineq_idx = genparams->hesssp_ineq_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, genparams->ngenON),
        RAJA_LAMBDA(RAJA::Index_type g) {
          if (gen_isrenewable[g])
            return;

          int gloc = gen_gineqidx[g];

          double lambda0 = lambdai_dev[gloc];
          double lambda1 = lambdai_dev[gloc + 1];
          double lsum = lambda0 + lambda1;

          int base = 3 * g;

          double v_pg_pg = 0.0;
          double v_pg_dev = -lsum;
          double v_pg_dpsys = gen_apf[g] * lsum;

          hess_dev[perm_dev[gen_hesssp_ineq_idx[base + 0]]] += v_pg_pg;
          hess_dev[perm_dev[gen_hesssp_ineq_idx[base + 1]]] += v_pg_dev;
          hess_dev[perm_dev[gen_hesssp_ineq_idx[base + 2]]] += v_pg_dpsys;
        });
  }

  // Set voltage inequality constraints Hessian (1 entry)
  if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
    int *bus_ispv = busparams->ispv_dev_;
    int *bus_isref = busparams->isref_dev_;
    int *bus_gineqidx = busparams->gineqidx_dev_;
    int *bus_genoffset = busparams->genoffset_dev_;
    int *bus_ngenONbus = busparams->ngenONbus_dev_;

    int *bus_hesssp_ineq_idx = busparams->hesssp_ineq_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type ibus) {
          if (!(bus_ispv[ibus] || bus_isref[ibus]))
            return;

          int gloc = bus_gineqidx[ibus];

          double lambda0 = lambdai_dev[gloc];
          double lambda1 = lambdai_dev[gloc + 1];
          double v = -(lambda0 + lambda1);

          int goff = bus_genoffset[ibus];
          int ngen = bus_ngenONbus[ibus];

          for (int k = 0; k < ngen; ++k) {
            int g = goff + k;

            hess_dev[perm_dev[bus_hesssp_ineq_idx[g]]] += v;
          }
        });
  }

  // Line inequality constraints Hessian (4x4, 10 upper triangular)
  {
    int *line_xidxf = lineparams->xidxf_dev_;
    int *line_xidxt = lineparams->xidxt_dev_;
    int *line_linelimidx = lineparams->linelimidx_dev_;
    int *line_isdcline = lineparams->isdcline_dev_;
    int *line_gineqidx = lineparams->gineqidx_dev_;
    int *line_hesssp_ineq_idx = lineparams->hesssp_ineq_idx_dev_;

    double *line_Gff = lineparams->Gff_dev_;
    double *line_Bff = lineparams->Bff_dev_;
    double *line_Gft = lineparams->Gft_dev_;
    double *line_Bft = lineparams->Bft_dev_;
    double *line_Gtf = lineparams->Gtf_dev_;
    double *line_Btf = lineparams->Btf_dev_;
    double *line_Gtt = lineparams->Gtt_dev_;
    double *line_Btt = lineparams->Btt_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, lineparams->nlinelim),
        RAJA_LAMBDA(RAJA::Index_type imon) {
          int iline = line_linelimidx[imon];
          if (line_isdcline[iline])
            return;

          int xlocf = line_xidxf[iline];
          int xloct = line_xidxt[iline];
          int base = 10 * imon;

          int gloc = line_gineqidx[imon];

          double lambdai_gloc = lambdai_dev[gloc];
          double lambdai_gloc1 = lambdai_dev[gloc + 1];

          double thetaf = x_dev[xlocf];
          double Vmf = x_dev[xlocf + 1];
          double thetat = x_dev[xloct];
          double Vmt = x_dev[xloct + 1];

          double thetaft = thetaf - thetat;
          double thetatf = thetat - thetaf;

          double sin_thetaft = sin(thetaft);
          double cos_thetaft = cos(thetaft);
          double sin_thetatf = sin(thetatf);
          double cos_thetatf = cos(thetatf);

          double Gff = line_Gff[iline];
          double Bff = line_Bff[iline];
          double Gft = line_Gft[iline];
          double Bft = line_Bft[iline];
          double Gtf = line_Gtf[iline];
          double Btf = line_Btf[iline];
          double Gtt = line_Gtt[iline];
          double Btt = line_Btt[iline];

          double Pf = Gff * Vmf * Vmf +
                      Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          double Qf = -Bff * Vmf * Vmf +
                      Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);

          double Pt = Gtt * Vmt * Vmt +
                      Vmt * Vmf * (Gtf * cos_thetatf + Btf * sin_thetatf);
          double Qt = -Btt * Vmt * Vmt +
                      Vmt * Vmf * (-Btf * cos_thetatf + Gtf * sin_thetatf);

          double dPf_dthetaf =
              Vmf * Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          double dPf_dVmf =
              2.0 * Gff * Vmf + Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          double dPf_dthetat =
              Vmf * Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          double dPf_dVmt = Vmf * (Gft * cos_thetaft + Bft * sin_thetaft);

          double dQf_dthetaf =
              Vmf * Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          double dQf_dVmf =
              -2.0 * Bff * Vmf + Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          double dQf_dthetat =
              Vmf * Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          double dQf_dVmt = Vmf * (-Bft * cos_thetaft + Gft * sin_thetaft);

          double dPt_dthetat =
              Vmt * Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          double dPt_dVmt =
              2.0 * Gtt * Vmt + Vmf * (Gtf * cos_thetatf + Btf * sin_thetatf);
          double dPt_dthetaf =
              Vmt * Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);
          double dPt_dVmf = Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);

          double dQt_dthetat =
              Vmt * Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);
          double dQt_dVmt =
              -2.0 * Btt * Vmt + Vmf * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          double dQt_dthetaf =
              Vmt * Vmf * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          double dQt_dVmf = Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);

          /* dPf_dthetaf = Vmf*Vmt*(-Gft*sin_thetaft + Bft*cos_thetaft); */
          double d2Pf_dthetaf_dthetaf =
              -Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          double d2Pf_dthetaf_dVmf =
              Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          double d2Pf_dthetaf_dthetat =
              Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          double d2Pf_dthetaf_dVmt =
              Vmf * (-Gft * sin_thetaft + Bft * cos_thetaft);

          /* dPf_Vmf  = 2*Gff*Vmf + Vmt*(Gft*cos_thetaft + Bft*sin_thetaft); */
          double d2Pf_dVmf_dVmf = 2.0 * Gff;
          double d2Pf_dVmf_dthetat =
              Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          double d2Pf_dVmf_dVmt = (Gft * cos_thetaft + Bft * sin_thetaft);

          /* dPf_dthetat = Vmf*Vmt*(Gft*sin_thetaft - Bft*cos_thetaft); */
          double d2Pf_dthetat_dthetat =
              Vmf * Vmt * (-Gft * cos_thetaft - Bft * sin_thetaft);
          double d2Pf_dthetat_dVmt =
              Vmf * (Gft * sin_thetaft - Bft * cos_thetaft);

          /* dPf_dVmt = Vmf*(Gft*cos_thetaft + Bft*sin_thetaft); */
          double d2Pf_dVmt_dVmt = 0.0;

          /* dQf_dthetaf = Vmf*Vmt*(Bft*sin_thetaft + Gft*cos_thetaft); */
          double d2Qf_dthetaf_dthetaf =
              Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          double d2Qf_dthetaf_dVmf =
              Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          double d2Qf_dthetaf_dthetat =
              Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          double d2Qf_dthetaf_dVmt =
              Vmf * (Bft * sin_thetaft + Gft * cos_thetaft);

          /* dQf_dVmf = -2*Bff*Vmf + Vmt*(-Bft*cos_thetaft + Gft*sin_thetaft);
           */
          double d2Qf_dVmf_dVmf = -2.0 * Bff;
          double d2Qf_dVmf_dthetat =
              Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          double d2Qf_dVmf_dVmt = (-Bft * cos_thetaft + Gft * sin_thetaft);

          /* dQf_dthetat = Vmf*Vmt*(-Bft*sin_thetaft - Gft*cos_thetaft); */
          double d2Qf_dthetat_dthetat =
              Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          double d2Qf_dthetat_dVmt =
              Vmf * (-Bft * sin_thetaft - Gft * cos_thetaft);

          /* dQf_dVmt = Vmf*(-Bft*cos_thetaft + Gft*sin_thetaft); */
          double d2Qf_dVmt_dVmt = 0.0;

          /* dPt_dthetat = Vmf*Vmt*(-Gtf*sin_thetatf + Btf*cos_thetatf); */
          double d2Pt_dthetat_dthetat =
              Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          double d2Pt_dthetat_dVmt =
              Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);

          /* dPt_Vmt  = 2*Gtt*Vmt + Vmf*(Gtf*cos_thetatf + Btf*sin_thetatf); */
          double d2Pt_dVmt_dVmt = 2.0 * Gtt;

          /* dPt_dthetaf = Vmf*Vmt*(Gtf*sin_thetatf - Btf*cos_thetatf); */
          double d2Pt_dthetaf_dthetat =
              Vmf * Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);
          double d2Pt_dthetaf_dVmt =
              Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);
          double d2Pt_dthetaf_dthetaf =
              Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          double d2Pt_dthetaf_dVmf =
              Vmt * (Gtf * sin_thetatf - Btf * cos_thetatf);

          /* dPt_dVmf = Vmt*(Gtf*cos_thetatf + Btf*sin_thetatf); */
          double d2Pt_dVmf_dthetat =
              Vmt * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          double d2Pt_dVmf_dVmt = (Gtf * cos_thetatf + Btf * sin_thetatf);
          double d2Pt_dVmf_dVmf = 0.0;

          /* dQt_dthetat = Vmf*Vmt*(Btf*sin_thetatf + Gtf*cos_thetatf); */
          double d2Qt_dthetat_dthetat =
              Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          double d2Qt_dthetat_dVmt =
              Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);

          /* dQt_dVmt = -2*Btt*Vmt + Vmf*(-Btf*cos_thetatf + Gtf*sin_thetatf);
           */
          double d2Qt_dVmt_dVmt = -2.0 * Btt;

          /* dQt_dthetaf = Vmf*Vmt*(-Btf*sin_thetatf - Gtf*cos_thetatf); */
          double d2Qt_dthetaf_dthetat =
              Vmf * Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          double d2Qt_dthetaf_dVmt =
              Vmf * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          double d2Qt_dthetaf_dthetaf =
              Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          double d2Qt_dthetaf_dVmf =
              Vmt * (-Btf * sin_thetatf - Gtf * cos_thetatf);

          /* dQt_dVmf = Vmt*(-Btf*cos_thetatf + Gtf*sin_thetatf); */
          double d2Qt_dVmf_dthetat =
              Vmt * (Btf * sin_thetatf + Gtf * cos_thetatf);
          double d2Qt_dVmf_dVmt = (-Btf * cos_thetatf + Gtf * sin_thetatf);
          double d2Qt_dVmf_dVmf = 0.0;

          double dSf2_dPf = 2.0 * Pf;
          double dSf2_dQf = 2.0 * Qf;
          double dSt2_dPt = 2.0 * Pt;
          double dSt2_dQt = 2.0 * Qt;

          double d2Sf2_00 = 2.0 * dPf_dthetaf * dPf_dthetaf +
                            dSf2_dPf * d2Pf_dthetaf_dthetaf +
                            2.0 * dQf_dthetaf * dQf_dthetaf +
                            dSf2_dQf * d2Qf_dthetaf_dthetaf;
          double d2Sf2_01 =
              2.0 * dPf_dVmf * dPf_dthetaf + dSf2_dPf * d2Pf_dthetaf_dVmf +
              2.0 * dQf_dVmf * dQf_dthetaf + dSf2_dQf * d2Qf_dthetaf_dVmf;
          double d2Sf2_02 = 2.0 * dPf_dthetat * dPf_dthetaf +
                            dSf2_dPf * d2Pf_dthetaf_dthetat +
                            2.0 * dQf_dthetat * dQf_dthetaf +
                            dSf2_dQf * d2Qf_dthetaf_dthetat;
          double d2Sf2_03 =
              2.0 * dPf_dVmt * dPf_dthetaf + dSf2_dPf * d2Pf_dthetaf_dVmt +
              2.0 * dQf_dVmt * dQf_dthetaf + dSf2_dQf * d2Qf_dthetaf_dVmt;

          double d2Sf2_11 =
              2.0 * dPf_dVmf * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dVmf +
              2.0 * dQf_dVmf * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dVmf;
          double d2Sf2_12 =
              2.0 * dPf_dthetat * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dthetat +
              2.0 * dQf_dthetat * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dthetat;
          double d2Sf2_13 =
              2.0 * dPf_dVmt * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dVmt +
              2.0 * dQf_dVmt * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dVmt;

          double d2Sf2_22 = 2.0 * dPf_dthetat * dPf_dthetat +
                            dSf2_dPf * d2Pf_dthetat_dthetat +
                            2.0 * dQf_dthetat * dQf_dthetat +
                            dSf2_dQf * d2Qf_dthetat_dthetat;
          double d2Sf2_23 =
              2.0 * dPf_dVmt * dPf_dthetat + dSf2_dPf * d2Pf_dthetat_dVmt +
              2.0 * dQf_dVmt * dQf_dthetat + dSf2_dQf * d2Qf_dthetat_dVmt;

          double d2Sf2_33 =
              2.0 * dPf_dVmt * dPf_dVmt + dSf2_dPf * d2Pf_dVmt_dVmt +
              2.0 * dQf_dVmt * dQf_dVmt + dSf2_dQf * d2Qf_dVmt_dVmt;

          double d2St2_00 = 2.0 * dPt_dthetaf * dPt_dthetaf +
                            dSt2_dPt * d2Pt_dthetaf_dthetaf +
                            2.0 * dQt_dthetaf * dQt_dthetaf +
                            dSt2_dQt * d2Qt_dthetaf_dthetaf;
          double d2St2_01 =
              2.0 * dPt_dVmf * dPt_dthetaf + dSt2_dPt * d2Pt_dthetaf_dVmf +
              2.0 * dQt_dVmf * dQt_dthetaf + dSt2_dQt * d2Qt_dthetaf_dVmf;
          double d2St2_02 = 2.0 * dPt_dthetat * dPt_dthetaf +
                            dSt2_dPt * d2Pt_dthetaf_dthetat +
                            2.0 * dQt_dthetat * dQt_dthetaf +
                            dSt2_dQt * d2Qt_dthetaf_dthetat;
          double d2St2_03 =
              2.0 * dPt_dVmt * dPt_dthetaf + dSt2_dPt * d2Pt_dthetaf_dVmt +
              2.0 * dQt_dVmt * dQt_dthetaf + dSt2_dQt * d2Qt_dthetaf_dVmt;

          double d2St2_11 =
              2.0 * dPt_dVmf * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dVmf +
              2.0 * dQt_dVmf * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dVmf;
          double d2St2_12 =
              2.0 * dPt_dthetat * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dthetat +
              2.0 * dQt_dthetat * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dthetat;
          double d2St2_13 =
              2.0 * dPt_dVmt * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dVmt +
              2.0 * dQt_dVmt * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dVmt;

          double d2St2_22 = 2.0 * dPt_dthetat * dPt_dthetat +
                            dSt2_dPt * d2Pt_dthetat_dthetat +
                            2.0 * dQt_dthetat * dQt_dthetat +
                            dSt2_dQt * d2Qt_dthetat_dthetat;
          double d2St2_23 =
              2.0 * dPt_dVmt * dPt_dthetat + dSt2_dPt * d2Pt_dthetat_dVmt +
              2.0 * dQt_dVmt * dQt_dthetat + dSt2_dQt * d2Qt_dthetat_dVmt;

          double d2St2_33 =
              2.0 * dPt_dVmt * dPt_dVmt + dSt2_dPt * d2Pt_dVmt_dVmt +
              2.0 * dQt_dVmt * dQt_dVmt + dSt2_dQt * d2Qt_dVmt_dVmt;

          double H00 = lambdai_gloc * d2Sf2_00 + lambdai_gloc1 * d2St2_00;
          double H01 = lambdai_gloc * d2Sf2_01 + lambdai_gloc1 * d2St2_01;
          double H02 = lambdai_gloc * d2Sf2_02 + lambdai_gloc1 * d2St2_02;
          double H03 = lambdai_gloc * d2Sf2_03 + lambdai_gloc1 * d2St2_03;
          double H11 = lambdai_gloc * d2Sf2_11 + lambdai_gloc1 * d2St2_11;
          double H12 = lambdai_gloc * d2Sf2_12 + lambdai_gloc1 * d2St2_12;
          double H13 = lambdai_gloc * d2Sf2_13 + lambdai_gloc1 * d2St2_13;
          double H22 = lambdai_gloc * d2Sf2_22 + lambdai_gloc1 * d2St2_22;
          double H23 = lambdai_gloc * d2Sf2_23 + lambdai_gloc1 * d2St2_23;
          double H33 = lambdai_gloc * d2Sf2_33 + lambdai_gloc1 * d2St2_33;

          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 0]]], H00);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 1]]], H01);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 2]]], H02);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 3]]], H03);

          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 4]]], H11);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 5]]], H12);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 6]]], H13);

          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 7]]], H22);
          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 8]]], H23);

          RAJA::atomicAdd<RAJA::auto_atomic>(
              &hess_dev[perm_dev[line_hesssp_ineq_idx[base + 9]]], H33);
        });
  }

  // Power-imbalance objective Hessian (2 diagonal entries)
  if (opflow->include_powerimbalance_variables) {
    int *bus_hesssp_obj_idx = busparams->hesssp_obj_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type ibus) {
          int base = 2 * ibus;
          hess_dev[perm_dev[bus_hesssp_obj_idx[base + 0]]] += 0.0;
          hess_dev[perm_dev[bus_hesssp_obj_idx[base + 1]]] += 0.0;
        });
  }

  // Gen objective Hessian (1 diagonal entry)
  {
    double obj_factor = opflow->obj_factor;
    double weight = opflow->weight;
    double MVAbase = ps->MVAbase;

    double *gen_cost_alpha = genparams->cost_alpha_dev_;
    int *gen_hesssp_obj_idx = genparams->hesssp_obj_idx_dev_;

    if (opflow->objectivetype == MIN_GEN_COST) {
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, genparams->ngenON),
          RAJA_LAMBDA(RAJA::Index_type igen) {
            double val = weight * obj_factor * 2.0 * gen_cost_alpha[igen] *
                         MVAbase * MVAbase;
            hess_dev[perm_dev[gen_hesssp_obj_idx[igen]]] += val;
          });
    } else if (opflow->objectivetype == MIN_GENSETPOINT_DEVIATION) {
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, genparams->ngenON),
          RAJA_LAMBDA(RAJA::Index_type igen) {
            double val = weight * obj_factor * 2.0;
            hess_dev[perm_dev[gen_hesssp_obj_idx[igen]]] += val;
          });
    }
  }

  // Load objective Hessian (2 diagonal entries)
  if (opflow->include_loadloss_variables) {
    int *load_hesssp_obj_idx = loadparams->hesssp_obj_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, loadparams->nload),
        RAJA_LAMBDA(RAJA::Index_type iload) {
          int base = 2 * iload;
          hess_dev[perm_dev[load_hesssp_obj_idx[base + 0]]] += 0.0;
          hess_dev[perm_dev[load_hesssp_obj_idx[base + 1]]] += 0.0;
        });
  }
}

#endif // EXAGO_ENABLE_HIOP_SPARSE
#endif // EXAGO_ENABLE_RAJA
