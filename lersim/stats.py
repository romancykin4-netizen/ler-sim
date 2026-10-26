"""Статистическая оценка результата с интервалами погрешности.

Отклонение края от номинального положения — случайная величина: на неё влияет дробовой шум химии
резиста и нестабильность параметров процесса. Точно её значение в конкретной точке предсказать нельзя,
но можно предсказать распределение: среднее (систематическую составляющую), разброс (случайную
составляющую) и интервал, в котором с заданной вероятностью окажется результат.

Для каждой величины выдаются:
  * среднее по N реализациям и доверительный интервал среднего  m ± t·s/√N  (сужается с ростом N);
  * интервал предсказания для отдельной линии  m ± t·s·√(1 + 1/N)  (не сужается: это разброс самих линий);
  * доля разброса, вносимая нестабильностью процесса, по сравнению с расчётом без неё на тех же
    случайных числах.
"""
from dataclasses import replace
from math import sqrt, log, pi, tan, atan, sin, cos, exp, lgamma
import numpy as np
from .process import simulate

QUANTITIES = (
    ("dev", "Отклонение края от номинального положения δ", "нм"),
    ("cd", "Ширина линии CD", "нм"),
    ("ler", "Шероховатость края LER (3σ)", "нм"),
    ("lwr", "Шероховатость ширины LWR (3σ)", "нм"),
)


def norm_ppf(p):
    """Обратная функция стандартного нормального распределения (алгоритм Акклама, погрешность ~1e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def t_cdf(t, df):
    """Функция распределения Стьюдента для целого ν в замкнутом виде (Абрамовиц, Стиган, 26.7.3–26.7.4)."""
    th = atan(abs(t) / sqrt(df))
    c2, s = cos(th) ** 2, sin(th)
    if df % 2:
        acc, term = 0.0, cos(th)
        for k in range(1, (df - 1) // 2 + 1):
            if k > 1:
                term *= c2 * (2 * k - 2) / (2 * k - 1)
            acc += term
        A = 2 / pi * (th + (s * acc if df > 1 else 0.0))
    else:
        acc, term = 1.0, 1.0
        for k in range(1, df // 2):
            term *= c2 * (2 * k - 1) / (2 * k)
            acc += term
        A = s * acc
    return 0.5 + 0.5 * A if t >= 0 else 0.5 - 0.5 * A


def t_ppf(p, df):
    """Квантиль Стьюдента: начальное приближение Корниша–Фишера, затем Ньютон по точной t_cdf."""
    if df == 1:
        return tan(pi * (p - 0.5))
    if df == 2:
        return (2 * p - 1) / sqrt(2 * p * (1 - p))
    z = norm_ppf(p)
    v = float(df)
    t = z + (z**3 + z) / (4 * v) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2) \
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    if df > 400:
        return t
    c = exp(lgamma((v + 1) / 2) - lgamma(v / 2)) / sqrt(v * pi)
    for _ in range(6):
        dt = (t_cdf(t, df) - p) / (c * (1 + t * t / v) ** (-(v + 1) / 2))
        t -= dt
        if abs(dt) < 1e-13 * max(1.0, abs(t)):
            break
    return t


def summary(x, conf):
    x = np.asarray(x, float)
    n = x.size
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    t = t_ppf(0.5 + conf / 2, n - 1) if n > 1 else float("nan")
    h_ci, h_pi = t * s / sqrt(n), t * s * sqrt(1 + 1 / n)
    return dict(n=n, mean=m, sd=s, ci=(m - h_ci, m + h_ci), pi=(m - h_pi, m + h_pi), t=t)


def per_line(r, p):
    """Величины каждой реализации и отклонения точек края от номинала (±CD₀/2)."""
    xL, xR = r["xL"], r["xR"]
    w = xR - xL
    uL, uR = xL - xL.mean(1, keepdims=True), xR - xR.mean(1, keepdims=True)
    sL, sR = np.sqrt((uL**2).mean(1)), np.sqrt((uR**2).mean(1))
    # Отклонение наружу считается положительным: край правее номинала справа или левее слева.
    dL, dR = -(xL + p.CD / 2), xR - p.CD / 2
    return dict(cd=w.mean(1), ler=3 * np.sqrt(0.5 * (sL**2 + sR**2)), lwr=3 * w.std(1),
                dev=0.5 * (dL.mean(1) + dR.mean(1)), points=np.concatenate([dL.ravel(), dR.ravel()]))


def perturbed(p, var, rng):
    """Случайные отклонения параметров процесса: доза и длина диффузии — в процентах, фокус и T — абсолютно."""
    kw = {}
    if var.get("dose", 0):
        kw["dose"] = p.dose * (1 + var["dose"] / 100 * rng.standard_normal())
    if var.get("focus", 0):
        kw["focus"] = p.focus + var["focus"] * rng.standard_normal()
    if var.get("T_peb", 0):
        kw["T_peb"] = p.T_peb + var["T_peb"] * rng.standard_normal()
    if var.get("L_D", 0):
        kw["L_D"] = max(1.0, p.L_D * (1 + var["L_D"] / 100 * rng.standard_normal()))
    return replace(p, **kw)


def estimate(p, n=32, conf=0.95, var=None, engine="auto", cancel=lambda: False, progress=lambda k, m: None):
    """Прогноз распределений величин для режима p.

    Без разброса параметров — один ансамблевый расчёт из n реализаций. С разбросом — n расчётов, в каждом
    свои отклонения параметров и тот же поток случайных чисел, что в расчёте без разброса (общие случайные
    числа): тогда разность дисперсий показывает вклад именно нестабильности процесса.
    """
    var = {k: v for k, v in (var or {}).items() if v}
    total = 2 * n if var else n
    base, noisy = [], []
    if not var:
        # Блоками, чтобы показывать ход расчёта и успевать остановить; блоки идут с разными генераторами.
        for i, k in enumerate(range(0, n, 8)):
            if cancel():
                return None
            base.append(per_line(simulate(replace(p, seed=p.seed + 1000 * i), min(8, n - k), engine, keep_map=False), p))
            progress(min(n, k + 8), total)
    else:
        rng = np.random.default_rng(p.seed + 7919)
        for i in range(n):
            if cancel():
                return None
            q = replace(p, seed=p.seed + 1000 * i)
            base.append(per_line(simulate(q, 1, engine, keep_map=False), p))
            noisy.append(per_line(simulate(perturbed(q, var, rng), 1, engine, keep_map=False), p))
            progress(2 * (i + 1), total)

    def join(rows):
        return {k: np.concatenate([np.atleast_1d(r[k]) for r in rows]) for k in rows[0]}

    b = join(base)
    t = join(noisy) if noisy else b
    out = dict(p=p, n=n, conf=conf, var=var, quantities={})
    for key, name, unit in QUANTITIES:
        sb, st = summary(b[key], conf), summary(t[key], conf)
        share = max(0.0, 1 - sb["sd"] ** 2 / st["sd"] ** 2) if var and st["sd"] > 0 else 0.0
        out["quantities"][key] = dict(name=name, unit=unit, stoch=sb, total=st, process_share=share, values=t[key])
    pts = t["points"]
    z = norm_ppf(0.5 + conf / 2)
    out["points"] = pts
    out["point_stats"] = dict(mean=float(pts.mean()), sd=float(pts.std()), mad=float(np.abs(pts - pts.mean()).mean()),
                              band=(float(pts.mean() - z * pts.std()), float(pts.mean() + z * pts.std())),
                              q=(float(np.quantile(pts, 0.5 - conf / 2)), float(np.quantile(pts, 0.5 + conf / 2))),
                              z=z, inside=float(np.mean(np.abs(pts - pts.mean()) <= z * pts.std())))
    return out
