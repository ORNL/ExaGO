
#include <exago_config.h>

#if defined(EXAGO_ENABLE_RAJA)
#if defined(EXAGO_ENABLE_HIOP_SPARSE)

#include <RAJA/RAJA.hpp>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <private/raja_exec_config.h>

#include "pbpolrajahiopsparse_gpu.hpp"

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
          double dPf_dthetaf =
              Vmf * Vmt * (-Gft * sin_ft + Bft * cos_ft);
          double dPf_dVmf =
              2.0 * Gff * Vmf + Vmt * (Gft * cos_ft + Bft * sin_ft);
          double dQf_dthetaf =
              Vmf * Vmt * (Bft * sin_ft + Gft * cos_ft);
          double dQf_dVmf =
              -2.0 * Bff * Vmf + Vmt * (-Bft * cos_ft + Gft * sin_ft);

          int pfbase = l_eqjacsp_diag[4 * l + 0];
          int qfbase = l_eqjacsp_diag[4 * l + 1];

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[pfbase + 0],
                                             dPf_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[pfbase + 1],
                                             dPf_dVmf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qfbase + 0],
                                             dQf_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qfbase + 1],
                                             dQf_dVmf);

          /* From-bus off-diagonal derivatives (atomicAdd for parallel lines) */
          double dPf_dthetat =
              Vmf * Vmt * (Gft * sin_ft - Bft * cos_ft);
          double dPf_dVmt = Vmf * (Gft * cos_ft + Bft * sin_ft);
          double dQf_dthetat =
              Vmf * Vmt * (-Bft * sin_ft - Gft * cos_ft);
          double dQf_dVmt = Vmf * (-Bft * cos_ft + Gft * sin_ft);

          int obase = l_eqjacsp_idx[l];
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 0],
                                             dPf_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 1],
                                             dPf_dVmt);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 2],
                                             dQf_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 3],
                                             dQf_dVmt);

          /* To-bus diagonal derivatives */
          double dPt_dthetat =
              Vmt * Vmf * (-Gtf * sin_tf + Btf * cos_tf);
          double dPt_dVmt =
              2.0 * Gtt * Vmt + Vmf * (Gtf * cos_tf + Btf * sin_tf);
          double dQt_dthetat =
              Vmt * Vmf * (Btf * sin_tf + Gtf * cos_tf);
          double dQt_dVmt =
              -2.0 * Btt * Vmt + Vmf * (-Btf * cos_tf + Gtf * sin_tf);

          int ptbase = l_eqjacsp_diag[4 * l + 2];
          int qtbase = l_eqjacsp_diag[4 * l + 3];

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[ptbase + 0],
                                             dPt_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[ptbase + 1],
                                             dPt_dVmt);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qtbase + 0],
                                             dQt_dthetat);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[qtbase + 1],
                                             dQt_dVmt);

          /* To-bus off-diagonal derivatives (atomicAdd for parallel lines) */
          double dPt_dthetaf =
              Vmt * Vmf * (Gtf * sin_tf - Btf * cos_tf);
          double dPt_dVmf = Vmt * (Gtf * cos_tf + Btf * sin_tf);
          double dQt_dthetaf =
              Vmt * Vmf * (-Btf * sin_tf - Gtf * cos_tf);
          double dQt_dVmf = Vmt * (-Btf * cos_tf + Gtf * sin_tf);

          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 4],
                                             dPt_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 5],
                                             dPt_dVmf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 6],
                                             dQt_dthetaf);
          RAJA::atomicAdd<RAJA::auto_atomic>(&jace_dev[obase + 7],
                                             dQt_dVmf);
        });
  }
}

#endif // EXAGO_ENABLE_HIOP_SPARSE
#endif // EXAGO_ENABLE_RAJA
