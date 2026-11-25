"""Modal estimate of oscillation damping (matrix pencil method)."""
from __future__ import annotations
import math
import numpy as np


def matrix_pencil(y, dt, rel_sv_tol=1e-3, max_order=20):
    """Fit y[k] ~ sum_i A_i z_i^k. Returns list of (freq_hz, zeta, amplitude)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    L = n // 3
    if L < 2:
        return []
    Y = np.array([y[i:i + L + 1] for i in range(n - L)])
    _, s, vh = np.linalg.svd(Y, full_matrices=False)
    m = int(min(max_order, max(1, (s > rel_sv_tol * s[0]).sum())))
    V = vh.conj().T[:, :m]
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1, :]) @ V[1:, :])
    z = z[np.abs(z) > 1e-12]
    lam = np.log(z) / dt
    Z = np.vander(z, n, increasing=True).T
    amp = np.linalg.lstsq(Z, y.astype(complex), rcond=None)[0]
    modes = []
    for l, a in zip(lam, amp):
        if l.imag <= 0:          # keep one of each conjugate pair
            continue
        w = abs(l)
        modes.append((l.imag / (2 * math.pi), -l.real / w, 2 * abs(a)))
    return modes


def damping_ratio_modal(t, x, t_start, f_min=0.1, f_max=2.0, fs=20.0, min_amplitude=1.0, t_ref=None):
    """Lowest damping ratio among electromechanical modes (f_min..f_max Hz) with
    amplitude >= min_amplitude in signal x(t), fitted from t_start on.
    The amplitude is referred back to t_ref (e.g. the clearing time; default
    t_start), so a mode that was large at clearing is not dropped only because
    it has decayed by t_start.
    Returns (zeta or None, modes) where modes = [(f, zeta, amp_at_t_ref), ...]."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    k = t >= t_start
    tt, xx = t[k], x[k]
    # Repeated time stamps (GridKit writes the event times twice) would make the
    # median step zero: keep the first sample of each time.
    tt, first = np.unique(tt, return_index=True)
    xx = xx[first]
    if len(tt) < 10:
        return None, []
    dt0 = float(np.median(np.diff(tt)))
    if not dt0 > 0:
        return None, []
    step = max(1, int(round((1.0 / fs) / dt0)))
    xx = xx[::step] - xx[::step].mean()
    dt = dt0 * step
    back = 0.0 if t_ref is None else max(0.0, tt[0] - t_ref)
    # Largest swing actually seen after t_ref (or t_start): a mode referred back
    # to t_ref can never be larger than that (caps spurious, heavily damped modes).
    kk = t >= (t_start if t_ref is None else t_ref)
    seen = float(np.max(np.abs(x[kk] - x[kk][int(0.8 * kk.sum()):].mean()))) if kk.any() else 0.0
    modes = []
    for f, z, a in matrix_pencil(xx, dt):
        if not (f_min <= f <= f_max) or abs(z) >= 1:
            continue
        sigma = z * 2 * math.pi * f / math.sqrt(1 - z * z)   # decay rate, 1/s
        a_ref = min(a * math.exp(sigma * back), seen)
        if a_ref >= min_amplitude:
            modes.append((f, z, a_ref))
    if not modes:
        return None, []
    return min(m[1] for m in modes), sorted(modes, key=lambda m: -m[2])


def damping_ratio_modal_robust(t, x, t_clear, windows=(1.0, 2.0), f_tol=0.05, zeta_tol=0.015,
                               f_min=0.1, f_max=2.0, fs=20.0, min_amplitude=1.0, threshold=None):
    """Matrix pencil in two windows (starting windows[0] and windows[1] s after
    clearing). A mode counts as confirmed if it is found in both windows with
    frequency within f_tol Hz and damping ratio within zeta_tol; its damping
    ratio is the mean of the two. A mode found in one window only is treated as
    a possible fit artefact.

    Returns (zeta, confirmed, status):
      zeta       lowest confirmed damping ratio, or None
      confirmed  [(f, zeta, amplitude_at_clearing), ...]
      status     "settled"         no mode >= min_amplitude in either window
                 "ok"              confirmed modes found; if *threshold* is given,
                                   every unconfirmed mode is also >= threshold
                 "above_threshold" no confirmed mode, but every fitted mode in
                                   both windows is >= threshold
                 "undetermined"    otherwise (an unconfirmed mode is below
                                   threshold, or no threshold given and nothing
                                   confirmed)
    """
    fits = [damping_ratio_modal(t, x, t_clear + w, f_min, f_max, fs, min_amplitude, t_ref=t_clear)[1]
            for w in windows]
    a, b = fits[0], fits[1]
    if not a and not b:
        return None, [], "settled"
    confirmed, used_a, used_b = [], set(), set()
    for i, (f1, z1, a1) in enumerate(a):
        # A mode of window b confirms at most one mode of window a.
        match = [(j, m) for j, m in enumerate(b)
                 if j not in used_b and abs(m[0] - f1) <= f_tol and abs(m[1] - z1) <= zeta_tol]
        if match:
            j, (f2, z2, a2) = min(match, key=lambda jm: abs(jm[1][0] - f1))
            used_a.add(i)
            used_b.add(j)
            confirmed.append(((f1 + f2) / 2, (z1 + z2) / 2, max(a1, a2)))
    unconfirmed = [m for i, m in enumerate(a) if i not in used_a] + [m for j, m in enumerate(b) if j not in used_b]
    low_unconfirmed = threshold is not None and any(m[1] < threshold for m in unconfirmed)
    confirmed.sort(key=lambda m: -m[2])
    zeta = min(m[1] for m in confirmed) if confirmed else None
    if confirmed and not low_unconfirmed:
        return zeta, confirmed, "ok"
    if not confirmed and threshold is not None and not low_unconfirmed:
        return None, [], "above_threshold"
    return zeta, confirmed, "undetermined"
