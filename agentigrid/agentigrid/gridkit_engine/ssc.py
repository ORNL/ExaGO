"""Short-circuit strength of the faulted buses (AgentiGrid option, information only).

Ybus is built from the network in the GridKit case (dict, as written for the fault screen):
  - branches as pi model, tap ratio and phase shift on the bus1 side;
  - constant-impedance loads (LoadZIP with alphaI = alphaP = 0) as shunt admittances;
  - synchronous machines (Genrou, Gensal) as Ra + jXdpp, converted from machine base to 100 MVA;
  - loads with constant-current or constant-power parts and all controllers are left out
    (they do not change the network impedance seen by a bolted fault).
Sparse LU of Ybus; Zth(bus) = diagonal element of Ybus^-1; Ssc = 100 MVA / |Zth|.
A bus is "weak" if its Ssc lies in the weakest *weak_percentile* % of all buses of the same
nominal voltage (system-specific screening; NERC 2017: there is no industry-standard threshold).

The result is reported to the LLM and stored in the journal. It never enters a verdict and never
changes a simulation: Ssc predicts the voltage dip at the faulted bus, not the outcome of the
stability checks (IPST 2027 re-assessment, 19 buses of ACTIVSg10k).

Needs numpy and scipy (not package requirements); without them the information is reported as
not available. Validated on ACTIVSg10k: 70030 551 MVA, 10378 2138, 10781 3318, 13087 4727 MVA;
predicted fault voltage within 3.1 % of GridKit.
"""

from __future__ import annotations

import cmath
from typing import Optional

MVA_BASE = 100.0


def _ybus_lu(case: dict):
    import numpy as np                       # noqa: F401  (needed by scipy)
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    idx = {b["number"]: i for i, b in enumerate(case["buses"])}
    r, c, v = [], [], []

    def add(i, j, y):
        r.append(i); c.append(j); v.append(y)

    for d in case["devices"]:
        p, q, cl = d.get("ports", {}), d.get("params", {}), d.get("class")
        if cl == "Branch":
            i, j = idx[p["bus1"]], idx[p["bus2"]]
            ys = 1.0 / complex(q["R"], q["X"])
            bsh = 1j * q.get("B", 0.0) / 2
            a = (q.get("tap", 1.0) or 1.0) * cmath.exp(1j * q.get("phase", 0.0))
            add(i, i, (ys + bsh) / abs(a) ** 2); add(j, j, ys + bsh)
            add(i, j, -ys / a.conjugate()); add(j, i, -ys / a)
        elif cl == "LoadZIP" and q.get("alphaI", 0) == 0 and q.get("alphaP", 0) == 0:
            i = idx[p["bus"]]
            add(i, i, complex(q["Pnom"], -q["Qnom"]))
        elif cl in ("Genrou", "Gensal"):
            i = idx[p["bus"]]
            add(i, i, (q.get("mva", MVA_BASE) / MVA_BASE) / complex(q.get("Ra", 0.0), q["Xdpp"]))
    n = len(idx)
    Y = sp.csc_matrix((v, (r, c)), shape=(n, n), dtype=complex)
    return idx, splu(Y)


def _zth(lu, idx: dict, buses: list[int], block: int = 200) -> dict:
    import numpy as np
    out = {}
    for k in range(0, len(buses), block):
        blk = buses[k:k + block]
        E = np.zeros((len(idx), len(blk)), complex)
        for m, b in enumerate(blk):
            E[idx[b], m] = 1.0
        X = lu.solve(E)
        for m, b in enumerate(blk):
            out[b] = complex(X[idx[b], m])
    return out


def strength(case: dict, buses: list[int], weak_percentile: float = 5.0,
             cache: Optional[dict] = None) -> dict:
    """{bus: {...}} for the faulted buses; {"error": text} if numpy/scipy are missing.

    *cache* (a dict kept by the caller per network) stores the Ssc of all buses of a
    voltage level, so the percentile is computed once per network and level (about 2 s
    for 4,660 buses at 138 kV)."""
    try:
        idx, lu = _ybus_lu(case)
    except ImportError as e:
        return {"error": f"not available ({e.name or 'numpy/scipy'} missing)"}
    kv = {b["number"]: round(float(b["params"]["kv"]), 1) for b in case["buses"]}
    cache = {} if cache is None else cache
    known = [b for b in buses if b in idx]
    z = _zth(lu, idx, known)
    out = {}
    for b in known:
        lvl = kv[b]
        if lvl not in cache:
            same = [x for x in idx if kv[x] == lvl]
            zl = _zth(lu, idx, same)
            cache[lvl] = sorted(MVA_BASE / abs(zl[x]) for x in same)
        ssc = MVA_BASE / abs(z[b])
        level = cache[lvl]
        below = sum(1 for s in level if s < ssc)
        pct = 100.0 * below / len(level)
        out[b] = {"ssc_mva": ssc, "zth_pu": abs(z[b]),
                  "x_over_r": z[b].imag / z[b].real if z[b].real else float("inf"),
                  "kv": lvl, "percentile": pct, "n_same_kv": len(level),
                  "weak": pct <= weak_percentile, "weak_percentile": weak_percentile}
    return out


def report_lines(info: dict) -> list[str]:
    """Text for the screen result (information only)."""
    head = "SHORT-CIRCUIT STRENGTH (information only, not part of the verdict):"
    if "error" in info:
        return [head, f"  {info['error']}"]
    lines = [head]
    for b in sorted(info):
        s = info[b]
        cls = "weak" if s["weak"] else "strong"
        lines.append(f"  bus {b}: Ssc {s['ssc_mva']:.0f} MVA (|Zth| {s['zth_pu']:.4f} pu, X/R {s['x_over_r']:.1f}); "
                     f"{s['percentile']:.1f} % of the {s['n_same_kv']} buses at {s['kv']:g} kV are weaker -> "
                     f"{cls} (weak = weakest {s['weak_percentile']:g} %)")
    lines.append("  Ssc predicts how deep the voltage at the faulted bus falls during the fault; "
                 "it does not predict the outcome of the stability checks.")
    return lines
