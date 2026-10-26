from dataclasses import replace
import numpy as np
from .process import aerial_image, grid, mean_field, edges_points, _lercore
from .core import edge_metrics


def cd_matrix(p, doses, focuses, cancel=lambda: False, progress=lambda k, n: None):
    """CD(доза, фокус) в пределе без шума (средние концентрации). Для проверок; окно процесса — window()."""
    x = grid(p)
    CD = np.full((doses.size, focuses.size), np.nan)
    n = CD.size
    for j, f in enumerate(focuses):
        I = aerial_image(x, p.CD, p.pitch, p.wl, p.NA, p.sigma_s, f)
        for i, d in enumerate(doses):
            if cancel():
                return None
            w, lost = mean_field(replace(p, dose=d, focus=f), I)
            CD[i, j] = np.nan if lost else w
            progress(j * doses.size + i + 1, n)
    return CD


def window(p, doses, focuses, engine="auto", cancel=lambda: False, progress=lambda k, n: None):
    """CD и LER на сетке доза × фокус полным стохастическим расчётом (одна реализация на точку).

    Во всех точках одни и те же случайные числа (общие случайные числа): молекулы ФГК и тушителя одни и те же,
    меняется только вероятность их превращения, поэтому кривые гладкие, а разности между точками отражают
    процесс, а не шум выборки. Точки с разрывом линии — NaN. Точки считаются пакетами, параллельно.
    """
    x = grid(p)
    D, F = np.meshgrid(doses, focuses, indexing="ij")
    D, F = D.ravel(), F.ravel()
    CD, LER = np.full(D.size, np.nan), np.full(D.size, np.nan)
    images = {f: aerial_image(x, p.CD, p.pitch, p.wl, p.NA, p.sigma_s, f) for f in focuses}
    batch = max(doses.size, 2 * (_lercore.threads() if _lercore is not None else 1))
    for k in range(0, D.size, batch):
        if cancel():
            return None
        sl = slice(k, min(D.size, k + batch))
        xL, xR, lost = edges_points(p, np.array([images[f] for f in F[sl]]), D[sl], engine)
        for i, j in enumerate(range(sl.start, sl.stop)):
            if not lost[i].any():
                m = edge_metrics(xL[i], xR[i])
                CD[j], LER[j] = m["CD"], 0.5 * (m["LER_L"] + m["LER_R"])
        progress(sl.stop, D.size)
    if _lercore is not None:
        _lercore.release()
    return CD.reshape(doses.size, focuses.size), LER.reshape(doses.size, focuses.size)


def dose_to_size(CD, doses, target):
    """Доза, дающая target в каждом столбце фокуса; CD(доза) монотонно убывает для позитивного резиста."""
    out = np.full(CD.shape[1], np.nan)
    for j in range(CD.shape[1]):
        c = CD[:, j]
        ok = np.isfinite(c)
        if ok.sum() > 1 and c[ok].min() <= target <= c[ok].max():
            out[j] = np.interp(target, c[ok][::-1], doses[ok][::-1])
    return out


def latitude(CD, doses, focuses, target, tol=0.1, el_min=5.0):
    """EL (%) по фокусу при допуске ±tol·CD и DOF при EL ≥ el_min, считая от дозы в номинале по размеру."""
    E0 = dose_to_size(CD, doses, target)
    EL = np.full(focuses.size, np.nan)
    for j in range(focuses.size):
        if np.isfinite(E0[j]):
            lo = dose_to_size(CD[:, [j]], doses, target * (1 + tol))[0]
            hi = dose_to_size(CD[:, [j]], doses, target * (1 - tol))[0]
            EL[j] = 100 * (hi - lo) / E0[j]
    inside = np.nan_to_num(EL) >= el_min
    dof = 0.0
    if inside.any():
        k = np.flatnonzero(inside)
        dof = focuses[k[-1]] - focuses[k[0]] if k.size > 1 else 0.0
    return E0, EL, dof
