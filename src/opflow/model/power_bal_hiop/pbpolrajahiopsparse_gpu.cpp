
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
    const int *bus_gidx = busparams->gidx_dev_;
    const double *bus_gl = busparams->gl_dev_;
    const double *bus_bl = busparams->bl_dev_;
    const int *bus_hesssp_eq_idx = busparams->hesssp_eq_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type ibus) {
          const int gloc = bus_gidx[ibus];

          const double lambdae_gloc = lambdae_dev[gloc];
          const double lambdae_gloc1 = lambdae_dev[gloc + 1];

          const double val = lambdae_gloc * (2.0 * bus_gl[ibus]) +
                             lambdae_gloc1 * (-2.0 * bus_bl[ibus]);

          hess_dev[perm_dev[bus_hesssp_eq_idx[ibus]]] += val;
        });
  }

  // Line equality constraints Hessian (4x4, 10 upper triangular)
  {
    const int *line_xidxf = lineparams->xidxf_dev_;
    const int *line_xidxt = lineparams->xidxt_dev_;
    const int *line_geqidxf = lineparams->geqidxf_dev_;
    const int *line_geqidxt = lineparams->geqidxt_dev_;
    const int *line_hesssp_eq_idx = lineparams->hesssp_eq_idx_dev_;

    const double *line_Gff = lineparams->Gff_dev_;
    const double *line_Bff = lineparams->Bff_dev_;
    const double *line_Gft = lineparams->Gft_dev_;
    const double *line_Bft = lineparams->Bft_dev_;
    const double *line_Gtf = lineparams->Gtf_dev_;
    const double *line_Btf = lineparams->Btf_dev_;
    const double *line_Gtt = lineparams->Gtt_dev_;
    const double *line_Btt = lineparams->Btt_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, lineparams->nlineON),
        RAJA_LAMBDA(RAJA::Index_type iline) {
          const int xlocf = line_xidxf[iline];
          const int xloct = line_xidxt[iline];
          const int base = 10 * iline;

          const int glocf = line_geqidxf[iline];
          const int gloct = line_geqidxt[iline];

          const double lambdaPf = lambdae_dev[glocf];
          const double lambdaQf = lambdae_dev[glocf + 1];
          const double lambdaPt = lambdae_dev[gloct];
          const double lambdaQt = lambdae_dev[gloct + 1];

          const double thetaf = x_dev[xlocf];
          const double Vmf = x_dev[xlocf + 1];
          const double thetat = x_dev[xloct];
          const double Vmt = x_dev[xloct + 1];

          const double thetaft = thetaf - thetat;
          const double thetatf = thetat - thetaf;

          const double sin_thetaft = sin(thetaft);
          const double cos_thetaft = cos(thetaft);
          const double sin_thetatf = sin(thetatf);
          const double cos_thetatf = cos(thetatf);

          const double Gff = line_Gff[iline];
          const double Bff = line_Bff[iline];
          const double Gft = line_Gft[iline];
          const double Bft = line_Bft[iline];
          const double Gtf = line_Gtf[iline];
          const double Btf = line_Btf[iline];
          const double Gtt = line_Gtt[iline];
          const double Btt = line_Btt[iline];

          const double dPf_dthetaf_dthetaf =
              -Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double dPf_dthetaf_dVmf =
              Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          const double dPf_dthetaf_dthetat =
              Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double dPf_dthetaf_dVmt =
              Vmf * (-Gft * sin_thetaft + Bft * cos_thetaft);

          const double dPf_dVmf_dVmf = 2.0 * Gff;
          const double dPf_dVmf_dthetat =
              Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          const double dPf_dVmf_dVmt = (Gft * cos_thetaft + Bft * sin_thetaft);

          const double dPf_dthetat_dthetat =
              Vmf * Vmt * (-Gft * cos_thetaft - Bft * sin_thetaft);
          const double dPf_dthetat_dVmt =
              Vmf * (Gft * sin_thetaft - Bft * cos_thetaft);

          const double dPf_dVmt_dVmt = 0.0;

          const double dQf_dthetaf_dthetaf =
              Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          const double dQf_dthetaf_dVmf =
              Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          const double dQf_dthetaf_dthetat =
              Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          const double dQf_dthetaf_dVmt =
              Vmf * (Bft * sin_thetaft + Gft * cos_thetaft);

          const double dQf_dVmf_dVmf = -2.0 * Bff;
          const double dQf_dVmf_dthetat =
              Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          const double dQf_dVmf_dVmt = (-Bft * cos_thetaft + Gft * sin_thetaft);

          const double dQf_dthetat_dthetat =
              Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          const double dQf_dthetat_dVmt =
              Vmf * (-Bft * sin_thetaft - Gft * cos_thetaft);

          const double dQf_dVmt_dVmt = 0.0;

          const double dPt_dthetaf_dthetaf =
              Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          const double dPt_dthetaf_dVmf =
              Vmt * (Gtf * sin_thetatf - Btf * cos_thetatf);
          const double dPt_dthetaf_dthetat =
              Vmf * Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double dPt_dthetaf_dVmt =
              Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);

          const double dPt_dVmf_dVmf = 0.0;
          const double dPt_dVmf_dthetat =
              Vmt * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          const double dPt_dVmf_dVmt = (Gtf * cos_thetatf + Btf * sin_thetatf);

          const double dPt_dthetat_dthetat =
              Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          const double dPt_dthetat_dVmf =
              Vmt * (-Gtf * sin_thetatf + Btf * cos_thetatf);

          const double dPt_dVmt_dVmt = 2.0 * Gtt;
          const double dPt_dVmt_dVmf = (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double dPt_dVmt_dthetat =
              Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);

          const double dQt_dthetaf_dthetaf =
              Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          const double dQt_dthetaf_dVmf =
              Vmt * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          const double dQt_dthetaf_dthetat =
              Vmf * Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          const double dQt_dthetaf_dVmt =
              Vmf * (-Btf * sin_thetatf - Gtf * cos_thetatf);

          const double dQt_dVmf_dVmf = 0.0;
          const double dQt_dVmf_dthetat =
              Vmt * (Btf * sin_thetatf + Gtf * cos_thetatf);
          const double dQt_dVmf_dVmt = (-Btf * cos_thetatf + Gtf * sin_thetatf);

          const double dQt_dthetat_dthetat =
              Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          const double dQt_dthetat_dVmf =
              Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);

          const double dQt_dVmt_dVmt = -2.0 * Btt;
          const double dQt_dVmt_dVmf = (Btf * sin_thetatf + Gtf * cos_thetatf);
          const double dQt_dVmt_dthetat =
              Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);

          const double H00 =
              lambdaPf * dPf_dthetaf_dthetaf + lambdaQf * dQf_dthetaf_dthetaf +
              lambdaPt * dPt_dthetaf_dthetaf + lambdaQt * dQt_dthetaf_dthetaf;
          const double H01 =
              lambdaPf * dPf_dthetaf_dVmf + lambdaQf * dQf_dthetaf_dVmf +
              lambdaPt * dPt_dthetaf_dVmf + lambdaQt * dQt_dthetaf_dVmf;
          const double H02 =
              lambdaPf * dPf_dthetaf_dthetat + lambdaQf * dQf_dthetaf_dthetat +
              lambdaPt * dPt_dthetaf_dthetat + lambdaQt * dQt_dthetaf_dthetat;
          const double H03 =
              lambdaPf * dPf_dthetaf_dVmt + lambdaQf * dQf_dthetaf_dVmt +
              lambdaPt * dPt_dthetaf_dVmt + lambdaQt * dQt_dthetaf_dVmt;

          const double H11 =
              lambdaPf * dPf_dVmf_dVmf + lambdaQf * dQf_dVmf_dVmf +
              lambdaPt * dPt_dVmf_dVmf + lambdaQt * dQt_dVmf_dVmf;
          const double H12 =
              lambdaPf * dPf_dVmf_dthetat + lambdaQf * dQf_dVmf_dthetat +
              lambdaPt * dPt_dVmf_dthetat + lambdaQt * dQt_dVmf_dthetat;
          const double H13 =
              lambdaPf * dPf_dVmf_dVmt + lambdaQf * dQf_dVmf_dVmt +
              lambdaPt * dPt_dVmf_dVmt + lambdaQt * dQt_dVmf_dVmt;

          const double H22 =
              lambdaPf * dPf_dthetat_dthetat + lambdaQf * dQf_dthetat_dthetat +
              lambdaPt * dPt_dthetat_dthetat + lambdaQt * dQt_dthetat_dthetat;
          const double H23 =
              lambdaPf * dPf_dthetat_dVmt + lambdaQf * dQf_dthetat_dVmt +
              lambdaPt * dPt_dthetat_dVmf + lambdaQt * dQt_dthetat_dVmf;

          const double H33 =
              lambdaPf * dPf_dVmt_dVmt + lambdaQf * dQf_dVmt_dVmt +
              lambdaPt * dPt_dVmt_dVmt + lambdaQt * dQt_dVmt_dVmt;

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
    const int *gen_isrenewable = genparams->isrenewable_dev_;
    const int *gen_gineqidx = genparams->gineqidxgen_dev_;
    const double *gen_apf = genparams->apf_dev_;
    const int *gen_hesssp_ineq_idx = genparams->hesssp_ineq_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, genparams->ngenON),
        RAJA_LAMBDA(RAJA::Index_type g) {
          if (gen_isrenewable[g])
            return;

          const int gloc = gen_gineqidx[g];

          const double lambdai_gloc = lambdai_dev[gloc];
          const double lambdai_gloc1 = lambdai_dev[gloc + 1];
          const double lsum = lambdai_gloc + lambdai_gloc1;

          const int base = 3 * g;

          const double v_pg_pg = 0.0;
          const double v_pg_dev = -lsum;
          const double v_pg_dpsys = gen_apf[g] * lsum;

          hess_dev[perm_dev[gen_hesssp_ineq_idx[base + 0]]] += v_pg_pg;
          hess_dev[perm_dev[gen_hesssp_ineq_idx[base + 1]]] += v_pg_dev;
          hess_dev[perm_dev[gen_hesssp_ineq_idx[base + 2]]] += v_pg_dpsys;
        });
  }

  // Set voltage inequality constraints Hessian (1 entry)
  if (opflow->genbusvoltagetype == FIXED_WITHIN_QBOUNDS) {
    const int *bus_ispv = busparams->ispv_dev_;
    const int *bus_isref = busparams->isref_dev_;
    const int *bus_gineqidx = busparams->gineqidx_dev_;
    const int *bus_genoffset = busparams->genoffset_dev_;
    const int *bus_ngenONbus = busparams->ngenONbus_dev_;

    const int *bus_hesssp_ineq_idx = busparams->hesssp_ineq_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type ibus) {
          if (!(bus_ispv[ibus] || bus_isref[ibus]))
            return;

          const int gloc = bus_gineqidx[ibus];

          const double lambdai_gloc = lambdai_dev[gloc];
          const double lambdai_gloc1 = lambdai_dev[gloc + 1];
          const double v = -(lambdai_gloc + lambdai_gloc1);

          const int goff = bus_genoffset[ibus];
          const int ngen = bus_ngenONbus[ibus];

          for (int k = 0; k < ngen; ++k) {
            const int g = goff + k;

            hess_dev[perm_dev[bus_hesssp_ineq_idx[g]]] += v;
          }
        });
  }

  // Line inequality constraints Hessian (4x4, 10 upper triangular)
  {
    const int *line_xidxf = lineparams->xidxf_dev_;
    const int *line_xidxt = lineparams->xidxt_dev_;
    const int *line_linelimidx = lineparams->linelimidx_dev_;
    const int *line_isdcline = lineparams->isdcline_dev_;
    const int *line_gineqidx = lineparams->gineqidx_dev_;
    const int *line_hesssp_ineq_idx = lineparams->hesssp_ineq_idx_dev_;

    const double *line_Gff = lineparams->Gff_dev_;
    const double *line_Bff = lineparams->Bff_dev_;
    const double *line_Gft = lineparams->Gft_dev_;
    const double *line_Bft = lineparams->Bft_dev_;
    const double *line_Gtf = lineparams->Gtf_dev_;
    const double *line_Btf = lineparams->Btf_dev_;
    const double *line_Gtt = lineparams->Gtt_dev_;
    const double *line_Btt = lineparams->Btt_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, lineparams->nlinelim),
        RAJA_LAMBDA(RAJA::Index_type imon) {
          const int iline = line_linelimidx[imon];
          if (line_isdcline[iline])
            return;

          const int xlocf = line_xidxf[iline];
          const int xloct = line_xidxt[iline];
          const int base = 10 * imon;

          const int gloc = line_gineqidx[imon];

          const double lambdai_gloc = lambdai_dev[gloc];
          const double lambdai_gloc1 = lambdai_dev[gloc + 1];

          const double thetaf = x_dev[xlocf];
          const double Vmf = x_dev[xlocf + 1];
          const double thetat = x_dev[xloct];
          const double Vmt = x_dev[xloct + 1];

          const double thetaft = thetaf - thetat;
          const double thetatf = thetat - thetaf;

          const double sin_thetaft = sin(thetaft);
          const double cos_thetaft = cos(thetaft);
          const double sin_thetatf = sin(thetatf);
          const double cos_thetatf = cos(thetatf);

          const double Gff = line_Gff[iline];
          const double Bff = line_Bff[iline];
          const double Gft = line_Gft[iline];
          const double Bft = line_Bft[iline];
          const double Gtf = line_Gtf[iline];
          const double Btf = line_Btf[iline];
          const double Gtt = line_Gtt[iline];
          const double Btt = line_Btt[iline];

          const double Pf = Gff * Vmf * Vmf +
                            Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double Qf =
              -Bff * Vmf * Vmf +
              Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);

          const double Pt = Gtt * Vmt * Vmt +
                            Vmt * Vmf * (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double Qt =
              -Btt * Vmt * Vmt +
              Vmt * Vmf * (-Btf * cos_thetatf + Gtf * sin_thetatf);

          const double dSf2_dPf = 2.0 * Pf;
          const double dSf2_dQf = 2.0 * Qf;
          const double dSt2_dPt = 2.0 * Pt;
          const double dSt2_dQt = 2.0 * Qt;

          const double dPf_dthetaf =
              Vmf * Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          const double dPf_dVmf =
              2.0 * Gff * Vmf + Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double dPf_dthetat =
              Vmf * Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          const double dPf_dVmt = Vmf * (Gft * cos_thetaft + Bft * sin_thetaft);

          const double dQf_dthetaf =
              Vmf * Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          const double dQf_dVmf =
              -2.0 * Bff * Vmf + Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          const double dQf_dthetat =
              Vmf * Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          const double dQf_dVmt =
              Vmf * (-Bft * cos_thetaft + Gft * sin_thetaft);

          const double dPt_dthetat =
              Vmt * Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          const double dPt_dVmt =
              2.0 * Gtt * Vmt + Vmf * (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double dPt_dthetaf =
              Vmt * Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);
          const double dPt_dVmf = Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);

          const double dQt_dthetat =
              Vmt * Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);
          const double dQt_dVmt =
              -2.0 * Btt * Vmt + Vmf * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          const double dQt_dthetaf =
              Vmt * Vmf * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          const double dQt_dVmf =
              Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);

          const double d2Pf_dthetaf_dthetaf =
              -Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double d2Pf_dthetaf_dVmf =
              Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          const double d2Pf_dthetaf_dthetat =
              Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double d2Pf_dthetaf_dVmt =
              Vmf * (-Gft * sin_thetaft + Bft * cos_thetaft);

          const double d2Pf_dVmf_dthetaf =
              Vmt * (-Gft * sin_thetaft + Bft * cos_thetaft);
          const double d2Pf_dVmf_dVmf = 2.0 * Gff;
          const double d2Pf_dVmf_dthetat =
              Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          const double d2Pf_dVmf_dVmt = (Gft * cos_thetaft + Bft * sin_thetaft);

          const double d2Pf_dthetat_dthetaf =
              Vmf * Vmt * (Gft * cos_thetaft + Bft * sin_thetaft);
          const double d2Pf_dthetat_dVmf =
              Vmt * (Gft * sin_thetaft - Bft * cos_thetaft);
          const double d2Pf_dthetat_dthetat =
              Vmf * Vmt * (-Gft * cos_thetaft - Bft * sin_thetaft);
          const double d2Pf_dthetat_dVmt =
              Vmf * (Gft * sin_thetaft - Bft * cos_thetaft);

          const double d2Pf_dVmt_dthetaf =
              Vmf * (-Gft * sin_thetaft + Bft * cos_thetaft);
          const double d2Pf_dVmt_dVmf = (Gft * cos_thetaft + Bft * sin_thetaft);
          const double d2Pf_dVmt_dthetat =
              Vmf * (Gft * sin_thetaft - Bft * cos_thetaft);
          const double d2Pf_dVmt_dVmt = 0.0;

          const double d2Qf_dthetaf_dthetaf =
              Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          const double d2Qf_dthetaf_dVmf =
              Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          const double d2Qf_dthetaf_dthetat =
              Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          const double d2Qf_dthetaf_dVmt =
              Vmf * (Bft * sin_thetaft + Gft * cos_thetaft);

          const double d2Qf_dVmf_dthetaf =
              Vmt * (Bft * sin_thetaft + Gft * cos_thetaft);
          const double d2Qf_dVmf_dVmf = -2.0 * Bff;
          const double d2Qf_dVmf_dthetat =
              Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          const double d2Qf_dVmf_dVmt =
              (-Bft * cos_thetaft + Gft * sin_thetaft);

          const double d2Qf_dthetat_dthetaf =
              Vmf * Vmt * (-Bft * cos_thetaft + Gft * sin_thetaft);
          const double d2Qf_dthetat_dVmf =
              Vmt * (-Bft * sin_thetaft - Gft * cos_thetaft);
          const double d2Qf_dthetat_dthetat =
              Vmf * Vmt * (Bft * cos_thetaft - Gft * sin_thetaft);
          const double d2Qf_dthetat_dVmt =
              Vmf * (-Bft * sin_thetaft - Gft * cos_thetaft);

          const double d2Qf_dVmt_dthetaf =
              Vmf * (Bft * sin_thetaft + Gft * cos_thetaft);
          const double d2Qf_dVmt_dVmf =
              (-Bft * cos_thetaft + Gft * sin_thetaft);
          const double d2Qf_dVmt_dthetat =
              Vmf * (-Bft * sin_thetaft - Gft * cos_thetaft);
          const double d2Qf_dVmt_dVmt = 0.0;

          const double d2Pt_dthetat_dthetat =
              Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          const double d2Pt_dthetat_dVmt =
              Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          const double d2Pt_dthetat_dthetaf =
              Vmf * Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double d2Pt_dthetat_dVmf =
              Vmt * (-Gtf * sin_thetatf + Btf * cos_thetatf);

          const double d2Pt_dVmt_dthetat =
              Vmf * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          const double d2Pt_dVmt_dVmt = 2.0 * Gtt;
          const double d2Pt_dVmt_dthetaf =
              Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);
          const double d2Pt_dVmt_dVmf = (Gtf * cos_thetatf + Btf * sin_thetatf);

          const double d2Pt_dthetaf_dthetat =
              Vmf * Vmt * (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double d2Pt_dthetaf_dVmt =
              Vmf * (Gtf * sin_thetatf - Btf * cos_thetatf);
          const double d2Pt_dthetaf_dthetaf =
              Vmf * Vmt * (-Gtf * cos_thetatf - Btf * sin_thetatf);
          const double d2Pt_dthetaf_dVmf =
              Vmt * (Gtf * sin_thetatf - Btf * cos_thetatf);

          const double d2Pt_dVmf_dthetat =
              Vmt * (-Gtf * sin_thetatf + Btf * cos_thetatf);
          const double d2Pt_dVmf_dVmt = (Gtf * cos_thetatf + Btf * sin_thetatf);
          const double d2Pt_dVmf_dthetaf =
              Vmt * (Gtf * sin_thetatf - Btf * cos_thetatf);
          const double d2Pt_dVmf_dVmf = 0.0;

          const double d2Qt_dthetat_dthetat =
              Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          const double d2Qt_dthetat_dVmt =
              Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);
          const double d2Qt_dthetat_dthetaf =
              Vmf * Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          const double d2Qt_dthetat_dVmf =
              Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);

          const double d2Qt_dVmt_dthetat =
              Vmf * (Btf * sin_thetatf + Gtf * cos_thetatf);
          const double d2Qt_dVmt_dVmt = -2.0 * Btt;
          const double d2Qt_dVmt_dthetaf =
              Vmf * (-Btf * sin_thetatf + Gtf * cos_thetatf);
          const double d2Qt_dVmt_dVmf =
              (-Btf * cos_thetatf + Gtf * sin_thetatf);

          const double d2Qt_dthetaf_dthetat =
              Vmf * Vmt * (-Btf * cos_thetatf + Gtf * sin_thetatf);
          const double d2Qt_dthetaf_dVmt =
              Vmf * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          const double d2Qt_dthetaf_dthetaf =
              Vmf * Vmt * (Btf * cos_thetatf - Gtf * sin_thetatf);
          const double d2Qt_dthetaf_dVmf =
              Vmt * (-Btf * sin_thetatf - Gtf * cos_thetatf);

          const double d2Qt_dVmf_dthetat =
              Vmt * (Btf * sin_thetatf + Gtf * cos_thetatf);
          const double d2Qt_dVmf_dVmt =
              (-Btf * cos_thetatf + Gtf * sin_thetatf);
          const double d2Qt_dVmf_dthetaf =
              Vmt * (-Btf * sin_thetatf - Gtf * cos_thetatf);
          const double d2Qt_dVmf_dVmf = 0.0;

          const double d2Sf2_dthetaf_dthetaf = 2.0 * dPf_dthetaf * dPf_dthetaf +
                                               dSf2_dPf * d2Pf_dthetaf_dthetaf +
                                               2.0 * dQf_dthetaf * dQf_dthetaf +
                                               dSf2_dQf * d2Qf_dthetaf_dthetaf;
          const double d2Sf2_dthetaf_dVmf =
              2.0 * dPf_dVmf * dPf_dthetaf + dSf2_dPf * d2Pf_dthetaf_dVmf +
              2.0 * dQf_dVmf * dQf_dthetaf + dSf2_dQf * d2Qf_dthetaf_dVmf;
          const double d2Sf2_dthetaf_dthetat = 2.0 * dPf_dthetat * dPf_dthetaf +
                                               dSf2_dPf * d2Pf_dthetaf_dthetat +
                                               2.0 * dQf_dthetat * dQf_dthetaf +
                                               dSf2_dQf * d2Qf_dthetaf_dthetat;
          const double d2Sf2_dthetaf_dVmt =
              2.0 * dPf_dVmt * dPf_dthetaf + dSf2_dPf * d2Pf_dthetaf_dVmt +
              2.0 * dQf_dVmt * dQf_dthetaf + dSf2_dQf * d2Qf_dthetaf_dVmt;

          const double d2Sf2_dVmf_dthetaf =
              2.0 * dPf_dthetaf * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dthetaf +
              2.0 * dQf_dthetaf * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dthetaf;
          const double d2Sf2_dVmf_dVmf =
              2.0 * dPf_dVmf * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dVmf +
              2.0 * dQf_dVmf * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dVmf;
          const double d2Sf2_dVmf_dthetat =
              2.0 * dPf_dthetat * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dthetat +
              2.0 * dQf_dthetat * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dthetat;
          const double d2Sf2_dVmf_dVmt =
              2.0 * dPf_dVmt * dPf_dVmf + dSf2_dPf * d2Pf_dVmf_dVmt +
              2.0 * dQf_dVmt * dQf_dVmf + dSf2_dQf * d2Qf_dVmf_dVmt;

          const double d2Sf2_dthetat_dthetaf = 2.0 * dPf_dthetaf * dPf_dthetat +
                                               dSf2_dPf * d2Pf_dthetat_dthetaf +
                                               2.0 * dQf_dthetat * dQf_dthetaf +
                                               dSf2_dQf * d2Qf_dthetat_dthetaf;
          const double d2Sf2_dthetat_dVmf =
              2.0 * dPf_dVmf * dPf_dthetat + dSf2_dPf * d2Pf_dthetat_dVmf +
              2.0 * dQf_dthetat * dQf_dVmf + dSf2_dQf * d2Qf_dthetat_dVmf;
          const double d2Sf2_dthetat_dthetat = 2.0 * dPf_dthetat * dPf_dthetat +
                                               dSf2_dPf * d2Pf_dthetat_dthetat +
                                               2.0 * dQf_dthetat * dQf_dthetat +
                                               dSf2_dQf * d2Qf_dthetat_dthetat;
          const double d2Sf2_dthetat_dVmt =
              2.0 * dPf_dVmt * dPf_dthetat + dSf2_dPf * d2Pf_dthetat_dVmt +
              2.0 * dQf_dthetat * dQf_dVmt + dSf2_dQf * d2Qf_dthetat_dVmt;

          const double d2Sf2_dVmt_dthetaf =
              2.0 * dPf_dthetaf * dPf_dVmt + dSf2_dPf * d2Pf_dVmt_dthetaf +
              2.0 * dQf_dthetaf * dQf_dVmt + dSf2_dQf * d2Qf_dVmt_dthetaf;
          const double d2Sf2_dVmt_dVmf =
              2.0 * dPf_dVmf * dPf_dVmt + dSf2_dPf * d2Pf_dVmt_dVmf +
              2.0 * dQf_dVmf * dQf_dVmt + dSf2_dQf * d2Qf_dVmt_dVmf;
          const double d2Sf2_dVmt_dthetat =
              2.0 * dPf_dthetat * dPf_dVmt + dSf2_dPf * d2Pf_dVmt_dthetat +
              2.0 * dQf_dthetat * dQf_dVmt + dSf2_dQf * d2Qf_dVmt_dthetat;
          const double d2Sf2_dVmt_dVmt =
              2.0 * dPf_dVmt * dPf_dVmt + dSf2_dPf * d2Pf_dVmt_dVmt +
              2.0 * dQf_dVmt * dQf_dVmt + dSf2_dQf * d2Qf_dVmt_dVmt;

          const double d2St2_dthetaf_dthetaf = 2.0 * dPt_dthetaf * dPt_dthetaf +
                                               dSt2_dPt * d2Pt_dthetaf_dthetaf +
                                               2.0 * dQt_dthetaf * dQt_dthetaf +
                                               dSt2_dQt * d2Qt_dthetaf_dthetaf;
          const double d2St2_dthetaf_dVmf =
              2.0 * dPt_dVmf * dPt_dthetaf + dSt2_dPt * d2Pt_dthetaf_dVmf +
              2.0 * dQt_dVmf * dQt_dthetaf + dSt2_dQt * d2Qt_dthetaf_dVmf;
          const double d2St2_dthetaf_dthetat = 2.0 * dPt_dthetat * dPt_dthetaf +
                                               dSt2_dPt * d2Pt_dthetaf_dthetat +
                                               2.0 * dQt_dthetat * dQt_dthetaf +
                                               dSt2_dQt * d2Qt_dthetaf_dthetat;
          const double d2St2_dthetaf_dVmt =
              2.0 * dPt_dVmt * dPt_dthetaf + dSt2_dPt * d2Pt_dthetaf_dVmt +
              2.0 * dQt_dVmt * dQt_dthetaf + dSt2_dQt * d2Qt_dthetaf_dVmt;

          const double d2St2_dVmf_dthetaf =
              2.0 * dPt_dthetaf * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dthetaf +
              2.0 * dQt_dthetaf * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dthetaf;
          const double d2St2_dVmf_dVmf =
              2.0 * dPt_dVmf * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dVmf +
              2.0 * dQt_dVmf * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dVmf;
          const double d2St2_dVmf_dthetat =
              2.0 * dPt_dthetat * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dthetat +
              2.0 * dQt_dthetat * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dthetat;
          const double d2St2_dVmf_dVmt =
              2.0 * dPt_dVmt * dPt_dVmf + dSt2_dPt * d2Pt_dVmf_dVmt +
              2.0 * dQt_dVmt * dQt_dVmf + dSt2_dQt * d2Qt_dVmf_dVmt;

          const double d2St2_dthetat_dthetaf = 2.0 * dPt_dthetaf * dPt_dthetat +
                                               dSt2_dPt * d2Pt_dthetat_dthetaf +
                                               2.0 * dQt_dthetaf * dQt_dthetat +
                                               dSt2_dQt * d2Qt_dthetat_dthetaf;
          const double d2St2_dthetat_dVmf =
              2.0 * dPt_dVmf * dPt_dthetat + dSt2_dPt * d2Pt_dthetat_dVmf +
              2.0 * dQt_dVmf * dQt_dthetat + dSt2_dQt * d2Qt_dthetat_dVmf;
          const double d2St2_dthetat_dthetat = 2.0 * dPt_dthetat * dPt_dthetat +
                                               dSt2_dPt * d2Pt_dthetat_dthetat +
                                               2.0 * dQt_dthetat * dQt_dthetat +
                                               dSt2_dQt * d2Qt_dthetat_dthetat;
          const double d2St2_dthetat_dVmt =
              2.0 * dPt_dVmt * dPt_dthetat + dSt2_dPt * d2Pt_dthetat_dVmt +
              2.0 * dQt_dVmt * dQt_dthetat + dSt2_dQt * d2Qt_dthetat_dVmt;

          const double d2St2_dVmt_dthetaf =
              2.0 * dPt_dthetaf * dPt_dVmt + dSt2_dPt * d2Pt_dVmt_dthetaf +
              2.0 * dQt_dthetaf * dQt_dVmt + dSt2_dQt * d2Qt_dVmt_dthetaf;
          const double d2St2_dVmt_dVmf =
              2.0 * dPt_dVmf * dPt_dVmt + dSt2_dPt * d2Pt_dVmt_dVmf +
              2.0 * dQt_dVmf * dQt_dVmt + dSt2_dQt * d2Qt_dVmt_dVmf;
          const double d2St2_dVmt_dthetat =
              2.0 * dPt_dthetat * dPt_dVmt + dSt2_dPt * d2Pt_dVmt_dthetat +
              2.0 * dQt_dthetat * dQt_dVmt + dSt2_dQt * d2Qt_dVmt_dthetat;
          const double d2St2_dVmt_dVmt =
              2.0 * dPt_dVmt * dPt_dVmt + dSt2_dPt * d2Pt_dVmt_dVmt +
              2.0 * dQt_dVmt * dQt_dVmt + dSt2_dQt * d2Qt_dVmt_dVmt;

          const double H00 = lambdai_gloc * d2Sf2_dthetaf_dthetaf +
                             lambdai_gloc1 * d2St2_dthetaf_dthetaf;
          const double H01 = lambdai_gloc * d2Sf2_dthetaf_dVmf +
                             lambdai_gloc1 * d2St2_dthetaf_dVmf;
          const double H02 = lambdai_gloc * d2Sf2_dthetaf_dthetat +
                             lambdai_gloc1 * d2St2_dthetaf_dthetat;
          const double H03 = lambdai_gloc * d2Sf2_dthetaf_dVmt +
                             lambdai_gloc1 * d2St2_dthetaf_dVmt;

          const double H11 =
              lambdai_gloc * d2Sf2_dVmf_dVmf + lambdai_gloc1 * d2St2_dVmf_dVmf;
          const double H12 = lambdai_gloc * d2Sf2_dVmf_dthetat +
                             lambdai_gloc1 * d2St2_dVmf_dthetat;
          const double H13 =
              lambdai_gloc * d2Sf2_dVmf_dVmt + lambdai_gloc1 * d2St2_dVmf_dVmt;

          const double H22 = lambdai_gloc * d2Sf2_dthetat_dthetat +
                             lambdai_gloc1 * d2St2_dthetat_dthetat;
          const double H23 = lambdai_gloc * d2Sf2_dthetat_dVmt +
                             lambdai_gloc1 * d2St2_dthetat_dVmt;

          const double H33 =
              lambdai_gloc * d2Sf2_dVmt_dVmt + lambdai_gloc1 * d2St2_dVmt_dVmt;

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
    const int *bus_hesssp_obj_idx = busparams->hesssp_obj_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, busparams->nbus),
        RAJA_LAMBDA(RAJA::Index_type ibus) {
          const int base = 2 * ibus;
          hess_dev[perm_dev[bus_hesssp_obj_idx[base + 0]]] += 0.0;
          hess_dev[perm_dev[bus_hesssp_obj_idx[base + 1]]] += 0.0;
        });
  }

  // Gen objective Hessian (1 diagonal entry)
  {
    const double obj_factor = opflow->obj_factor;
    const double weight = opflow->weight;
    const double MVAbase = ps->MVAbase;

    const double *gen_cost_alpha = genparams->cost_alpha_dev_;
    const int *gen_hesssp_obj_idx = genparams->hesssp_obj_idx_dev_;

    if (opflow->objectivetype == MIN_GEN_COST) {
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, genparams->ngenON),
          RAJA_LAMBDA(RAJA::Index_type igen) {
            const double val = weight * obj_factor * 2.0 *
                               gen_cost_alpha[igen] * MVAbase * MVAbase;
            hess_dev[perm_dev[gen_hesssp_obj_idx[igen]]] += val;
          });
    } else if (opflow->objectivetype == MIN_GENSETPOINT_DEVIATION) {
      RAJA::forall<exago_raja_exec>(
          RAJA::RangeSegment(0, genparams->ngenON),
          RAJA_LAMBDA(RAJA::Index_type igen) {
            const double val = weight * obj_factor * 2.0;
            hess_dev[perm_dev[gen_hesssp_obj_idx[igen]]] += val;
          });
    }
  }

  // Load objective Hessian (2 diagonal entries)
  if (opflow->include_loadloss_variables) {
    const int *load_hesssp_obj_idx = loadparams->hesssp_obj_idx_dev_;

    RAJA::forall<exago_raja_exec>(
        RAJA::RangeSegment(0, loadparams->nload),
        RAJA_LAMBDA(RAJA::Index_type iload) {
          const int base = 2 * iload;
          hess_dev[perm_dev[load_hesssp_obj_idx[base + 0]]] += 0.0;
          hess_dev[perm_dev[load_hesssp_obj_idx[base + 1]]] += 0.0;
        });
  }
}

#endif // EXAGO_ENABLE_HIOP_SPARSE
#endif // EXAGO_ENABLE_RAJA
