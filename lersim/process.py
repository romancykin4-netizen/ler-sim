"""Сквозная цепочка O03–O05 → P05 → R02/R05/R06/R09 → D01/D03 для решётки линия/пробел.

Единицы модуля: нм, с, К, эВ, мДж/см², концентрации в нм^-3 (спектральное ядро core от единиц не зависит).
Значения резиста демонстрационные, без калибровки. Стохастика вносится только экспозицией (ветвь 2
спецификации: биномиальное превращение конечной популяции PAG); PEB детерминирован и переносит её.
"""
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, fields
import numpy as np
from .core import arrhenius, development_rate, detrend, pair_spectra, edge_metrics


@dataclass(frozen=True)
class Process:
    CD: float = 120.0         # нм, ширина тёмной линии на маске в масштабе пластины
    pitch: float = 300.0
    wl: float = 193.0
    NA: float = 0.75
    sigma_s: float = 0.6      # частичная когерентность, круглый источник
    dose: float = 32.5        # мДж/см², близко к дозе номинального размера для CD₀ = 120 нм
    focus: float = 0.0
    T_peb: float = 105.0      # °C
    t_peb: float = 60.0
    t_dev: float = 60.0
    L: float = 1200.0
    dx: float = 1.0           # поперёк линии
    dy: float = 1.0           # вдоль линии
    seed: int = 1
    c_pag: float = 0.12       # нм^-3
    c_q: float = 0.03
    C_g: float = 0.03         # см²/мДж
    h_eff: float = 50.0       # нм, глубина усреднения ячейки при виде сверху
    L_D: float = 15.5         # нм, длина диффузии кислоты √(2·D·t) при T_ref и t_ref
    t_ref: float = 60.0       # с
    kd_ref: float = 1.0       # нм³/с
    T_ref: float = 105.0
    E_D: float = 0.5          # эВ
    E_kd: float = 0.8
    R_max: float = 100.0      # нм/с
    R_min: float = 0.02
    n_dev: float = 8.0
    d_th: float = 0.545       # пороговая степень депротекции модели Мака
    r_c: float = 1.5          # нм, радиус реакции: размер облака молекулы кислоты и тушителя

    @property
    def a_dev(self):
        """Параметр a модели Мака через порог: a = (n+1)/(n−1)·d_th^n (Mack, 1987)."""
        return (self.n_dev + 1) / (self.n_dev - 1) * self.d_th**self.n_dev

    @property
    def D_ref(self):
        """Коэффициент диффузии при T_ref из заданной длины диффузии: L_D² = 2·D·t_ref."""
        return self.L_D**2 / (2 * self.t_ref)

    @classmethod
    def from_dict(cls, d):
        """Параметры из файла; старые файлы задавали D_ref (нм²/с) вместо длины диффузии."""
        d = dict(d)
        if "a_dev" in d and "d_th" not in d:
            n = d.get("n_dev", 8.0)
            d["d_th"] = float((d["a_dev"] * (n - 1) / (n + 1)) ** (1 / n))
        if "D_ref" in d and "L_D" not in d:
            d["L_D"] = float(np.sqrt(2 * d["D_ref"] * d.get("t_ref", 60.0)))
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


_IMG = OrderedDict()


def aerial_image(x, CD, pitch, wl, NA, sigma_s, focus, n_src=15):
    """Кэш последних изображений: при изменении дозы, PEB, проявления оптика не пересчитывается."""
    key = (x.size, float(x[0]), float(x[1] - x[0]) if x.size > 1 else 0.0, CD, pitch, wl, NA, sigma_s, focus, n_src)
    I = _IMG.get(key)
    if I is None:
        I = _aerial_image(x, CD, pitch, wl, NA, sigma_s, focus, n_src)
        I.setflags(write=False)
        _IMG[key] = I
        if len(_IMG) > 256:
            _IMG.popitem(last=False)
    else:
        _IMG.move_to_end(key)
    return I


def _aerial_image(x, CD, pitch, wl, NA, sigma_s, focus, n_src=15):
    """O03–O05: нормировка по открытой маске, I(open frame) = 1."""
    fc = NA / wl
    nmax = int(np.ceil((1 + sigma_s) * fc * pitch)) + 1
    n = np.arange(-nmax, nmax + 1)
    fn = n / pitch
    duty = CD / pitch
    m = np.where(n == 0, 1 - duty, -duty * np.sinc(n * duty))
    if sigma_s > 0:
        g = np.linspace(-1, 1, n_src)
        sx, sy = np.meshgrid(g, g)
        inside = sx**2 + sy**2 <= 1
        sx, sy = sx[inside] * sigma_s * fc, sy[inside] * sigma_s * fc
    else:
        sx = sy = np.zeros(1)
    r2 = (fn[None, :] + sx[:, None]) ** 2 + sy[:, None] ** 2
    kz = 2 * np.pi * np.sqrt(np.maximum(wl**-2 - r2, 0))
    kz0 = 2 * np.pi * np.sqrt(np.maximum(wl**-2 - sx**2 - sy**2, 0))[:, None]
    # Дефокус берётся относительно наклонной плоской волны своей точки источника: общий фазовый
    # множитель точки источника на интенсивность не влияет (O03).
    P = (r2 <= fc**2) * np.exp(1j * focus * (kz - kz0))
    A = (m * P) @ np.exp(2j * np.pi * np.outer(fn, x))
    return (np.abs(A) ** 2).mean(0)


def expose(I, p, rng, ny):
    """P05 с пуассоновским числом PAG и основания в ячейке (P08, некоррелированный состав)."""
    V = p.dx * p.dy * p.h_eff
    pG = -np.expm1(-p.C_g * p.dose * I)
    NG = rng.poisson(p.c_pag * V, (ny, I.size))
    NH = rng.binomial(NG, np.broadcast_to(pG, NG.shape))
    NQ = rng.poisson(p.c_q * V, NG.shape)
    return NH / V, NQ / V


def peb(cH, cQ, p, n_steps=30):
    """R03–R06 с расщеплением: диффузия → нейтрализация → накопление ∫[H]dt (R09, точное спектральное решение).

    Молекулы кислоты и тушителя — не точки, а облака радиуса r_c (радиус реакции): без этого скорость
    захвата кислоты тушителем и депротекция у места рождения кислоты зависели бы от размера ячейки
    (в 2D интеграл функции Грина по времени расходится логарифмически). Тушитель неподвижен (D_Q = 0),
    нейтрализация мгновенная: за шаг кислота и тушитель взаимно уничтожаются до исчерпания меньшего.
    ∫[H]dt — по формуле трапеций (второй порядок по шагу).
    """
    T, Tr = p.T_peb + 273.15, p.T_ref + 273.15
    D = arrhenius(p.D_ref, p.E_D, T, Tr)
    kd = arrhenius(p.kd_ref, p.E_kd, T, Tr)
    dt = p.t_peb / n_steps
    fy = np.fft.fftfreq(cH.shape[0], p.dy)[:, None]
    fx = np.fft.rfftfreq(cH.shape[1], p.dx)[None, :]
    k2 = (2 * np.pi) ** 2 * (fx**2 + fy**2)
    blur = lambda c, G: np.maximum(np.fft.irfft2(np.fft.rfft2(c) * G, s=c.shape), 0)
    if p.r_c > 0:
        G0 = np.exp(-0.5 * p.r_c**2 * k2)
        cH, cQ = blur(cH, G0), blur(cQ, G0)
    G = np.exp(-D * k2 * dt)
    q = np.minimum(cH, cQ)
    cH, cQ = cH - q, cQ - q
    act = 0.5 * dt * cH
    for k in range(1, n_steps + 1):
        cH = blur(cH, G)
        q = np.minimum(cH, cQ)
        cH, cQ = cH - q, cQ - q
        act += (0.5 if k == n_steps else 1.0) * dt * cH
    return -np.expm1(-kd * act), D, kd


def _godunov(a, b, ha, hb):
    """Разностное уравнение эйконала ((T−a)/ha)² + ((T−b)/hb)² = 1 (схема Годунова, первый порядок).

    ha = Δx·f̄, hb = Δy·f̄ — время пересечения ребра со средней медленностью f = 1/R двух ячеек: в одномерном
    пределе это точное время пути через кусочно-постоянную среду. Если двусторонний корень меньше
    наибольшего из соседей, фронт приходит с одной стороны и T = min(a + ha, b + hb).
    """
    t1 = np.minimum(a + ha, b + hb)
    wa, wb = 1 / ha**2, 1 / hb**2
    s = wa + wb
    m = (wa * a + wb * b) / s
    disc = m * m - (wa * a * a + wb * b * b - 1) / s
    t2 = m + np.sqrt(np.maximum(disc, 0))
    return np.where((t1 > np.maximum(a, b)) & (disc >= 0), np.minimum(t1, t2), t1)


def arrival(R, dx, dy, t_max, tol=1e-9, itmax=500):
    """D03: время прихода фронта проявления T(x, y), |∇T| = 1/R, источник — центр пробела (x = ±p/2).

    Быстрые проходы (Чжао, 2005) поочерёдно по столбцам и строкам до неподвижной точки разностной схемы.
    Фронт обходит медленные ячейки сбоку, поэтому край формируется путями в плоскости x–y, а не по прямой
    в каждом сечении. Значения выше t_max на положение края не влияют и в критерии сходимости не участвуют.
    """
    ny, nx = R.shape
    f = 1 / R
    T = np.full((ny, nx), 1e30)
    T[:, 0], T[:, -1] = 0.5 * dx * f[:, 0], 0.5 * dx * f[:, -1]
    up, dn = np.r_[ny - 1, :ny - 1], np.r_[1:ny, 0]
    for _ in range(itmax):
        T0 = np.minimum(T, t_max)
        for cols in (range(1, nx - 1), range(nx - 2, 0, -1)):
            for i in cols:
                l, r, u, w = T[:, i - 1], T[:, i + 1], T[up, i], T[dn, i]
                fa = np.where(l <= r, f[:, i - 1], f[:, i + 1])
                fb = np.where(u <= w, f[up, i], f[dn, i])
                c = f[:, i]
                T[:, i] = np.minimum(T[:, i], _godunov(np.minimum(l, r), np.minimum(u, w), 0.5 * dx * (c + fa),
                                                       0.5 * dy * (c + fb)))
        for rows in (range(ny), range(ny - 1, -1, -1)):
            for j in rows:
                l, r, u, w = T[j, :-2], T[j, 2:], T[up[j], 1:-1], T[dn[j], 1:-1]
                fa = np.where(l <= r, f[j, :-2], f[j, 2:])
                fb = np.where(u <= w, f[up[j], 1:-1], f[dn[j], 1:-1])
                c = f[j, 1:-1]
                T[j, 1:-1] = np.minimum(T[j, 1:-1], _godunov(np.minimum(l, r), np.minimum(u, w), 0.5 * dx * (c + fa),
                                                             0.5 * dy * (c + fb)))
        if np.abs(np.minimum(T, t_max) - T0).max() <= tol * t_max:
            break
    return T


def edges(T, R, x, t_dev):
    """Края — первые проявленные ячейки при движении от центра линии наружу; положение внутри шага —
    по кусочно-постоянной скорости (полшага со скоростью своей ячейки, полшага — соседней)."""
    ny, nx = T.shape
    dx = x[1] - x[0]
    c = nx // 2
    f = 1 / R
    rows = np.arange(ny)
    lost = (T[:, c - 1] <= t_dev) | (T[:, c] <= t_dev)

    def place(i, k, sgn):
        e = t_dev - T[rows, i]
        fi, fk = f[rows, i], f[rows, k]
        return x[i] + sgn * np.where(e < 0.5 * dx * fi, e / fi, 0.5 * dx + (e - 0.5 * dx * fi) / fk)

    dev = T <= t_dev
    # Если пробел не проявлен даже у источника, формула ниже даёт край на расстоянии R·t_dev от центра пробела.
    iL = np.where(dev[:, :c].any(1), c - 1 - np.argmax(dev[:, :c][:, ::-1], axis=1), 0)
    iR = np.where(dev[:, c:].any(1), c + np.argmax(dev[:, c:], axis=1), nx - 1)
    iL, iR = np.clip(iL, 0, c - 2), np.clip(iR, c + 1, nx - 1)
    xl, xr = place(iL, iL + 1, 1.0), place(iR, iR - 1, -1.0)
    return np.where(lost, 0.0, xl), np.where(lost, 0.0, xr), lost


def develop(d, x, p):
    """D01 + D03: скорость Мака в каждой ячейке, двумерный фронт от центров пробелов, края по t_dev.

    Вертикальное проявление (по толщине) и ограничение подачи проявителя (D06) не учитываются.
    """
    R = development_rate(d, p.R_min, p.R_max, p.a_dev, p.n_dev)
    T = arrival(R, p.dx, p.dy, p.t_dev)
    return edges(T, R, x, p.t_dev)


try:
    from . import _lercore
except ImportError:
    _lercore = None


def gauss_kernels(p, s):
    """Одномерные гауссовы ядра со стандартным отклонением s в узлах сетки, нормированные на 1.

    Хвост за 4,5σ — 7e-6 массы; ошибка передаточной функции 3e-6, меньше ошибки дискретизации сетки.
    """
    def k1(h):
        if s < 1e-6 * h:
            return np.ones(1)
        r = max(1, int(np.ceil(4.5 * s / h)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) * h / s) ** 2)
        return k / k.sum()
    return k1(p.dx), k1(p.dy)


def diffusion_kernels(p, D, dt):
    """Ядра шага диффузии: тепловое ядро с σ² = 2·D·dt. Двумерное ядро разделимо: свёртка по x, затем по y."""
    return gauss_kernels(p, np.sqrt(2 * D * dt))


def simulate(p, n_real=1, engine="auto", n_steps=30, keep_map=True):
    """Полная стохастическая цепочка. engine: "auto" (C++ при наличии), "cpp", "numpy".

    Возвращает также stages — список (этап, сводка, секунды) для журнала.
    keep_map=False не хранит карту депротекции (для серийных расчётов, где нужны только метрики).
    """
    use_cpp = engine == "cpp" or (engine == "auto" and _lercore is not None)
    stages, t0 = [], time.perf_counter()
    clock = [t0]

    def lap(name, text):
        now = time.perf_counter()
        stages.append((name, text, now - clock[0]))
        clock[0] = now

    x = grid(p)
    nx, ny = x.size, int(round(p.L / p.dy))
    y = np.arange(ny) * p.dy
    I = aerial_image(x, p.CD, p.pitch, p.wl, p.NA, p.sigma_s, p.focus)
    i_e = np.searchsorted(x, p.CD / 2)
    nils = p.CD * abs(np.gradient(np.log(I), p.dx)[i_e])
    lap("Изображение", f"интенсивность от {I.min():.3f} до {I.max():.3f}, контраст "
        f"{(I.max()-I.min())/(I.max()+I.min()):.3f}, нормированный наклон на заданном крае (NILS) {nils:.2f}")

    V = p.dx * p.dy * p.h_eff
    pG = -np.expm1(-p.C_g * p.dose * I)
    T, Tr = p.T_peb + 273.15, p.T_ref + 273.15
    D = arrhenius(p.D_ref, p.E_D, T, Tr)
    kd = arrhenius(p.kd_ref, p.E_kd, T, Tr)
    dt = p.t_peb / n_steps
    lap("Экспонирование", f"ячейка {p.dx:g}×{p.dy:g}×{p.h_eff:g} нм, в среднем {p.c_pag*V:.1f} молекул генератора кислоты и "
        f"{p.c_q*V:.1f} молекул тушителя на ячейку; доля разложенного генератора на крае {pG[i_e]:.3f}, "
        f"в центре пробела {pG.max():.3f}")

    if use_cpp:
        kx, ky = diffusion_kernels(p, D, dt)
        kx0, ky0 = gauss_kernels(p, p.r_c)
        q = dict(n_steps=n_steps, dx=p.dx, dy=p.dy, V=V, c_pag=p.c_pag, c_q=p.c_q, dt=dt, kd=kd, t_dev=p.t_dev,
                 R_min=p.R_min, R_max=p.R_max, a_dev=p.a_dev, n_dev=p.n_dev, pitch=p.pitch)
        xL, xR, lost, d0, acid0 = _lercore.run(pG, kx, ky, kx0, ky0, ny, n_real, p.seed, q, keep_map)
        lost = lost.astype(bool)
        how = f"C++, потоков: {_lercore.threads()}, ядро диффузии {kx.size}×{ky.size} узлов"
    else:
        rng = np.random.default_rng(p.seed)
        xL, xR, lost = [], [], []
        for i in range(n_real):
            cH, cQ = expose(I, p, rng, ny)
            d = peb(cH, cQ, p, n_steps)[0]
            l, r, b = develop(d, x, p)
            xL.append(l); xR.append(r); lost.append(b)
            if i == 0:
                d0, acid0 = d, cH.mean(0)
        xL, xR, lost = np.array(xL), np.array(xR), np.array(lost)
        how = "NumPy"
    L_diff = np.sqrt(2 * D * p.t_peb)
    lap("Термообработка и проявление", f"повторов {n_real} ({how}); длина диффузии кислоты задана {p.L_D:g} нм "
        f"при {p.T_ref:g} °C и {p.t_ref:g} с, при {p.T_peb:g} °C и {p.t_peb:g} с — {L_diff:.1f} нм (D = {D:.2f} нм²/с), константа депротекции {kd:.2f} нм³/с; "
        f"{n_steps} шагов по {dt:.2f} с")

    m = edge_metrics(xL, xR)
    uL, uR = detrend(xL)[0], detrend(xR)[0]
    sp = pair_spectra(uL, uR, p.dy)
    S_edge = 0.5 * (sp["LL"] + sp["RR"]).mean(0)
    lap("Обработка краёв", f"ширина {m['CD']:.2f} нм, σ левого/правого края {m['sigma_L']:.3f}/{m['sigma_R']:.3f} нм, "
        f"σ ширины {m['sigma_w']:.3f} нм, корреляция краёв {m['rho']:+.3f}")
    if not keep_map:
        d0 = None
    elif d0.dtype != np.float32:
        d0 = d0.astype(np.float32)
    return dict(x=x, y=y, I=I, d=d0, d_mean=None if d0 is None else d0.mean(0, dtype=np.float64), acid0=acid0, xL=xL, xR=xR,
                lost_frac=lost.mean(), f=sp["f"], S_edge=S_edge, S_w=sp["w"].mean(0),
                m=m, NILS=nils, D=D, kd=kd, L_diff=L_diff, engine="C++" if use_cpp else "NumPy",
                sigma_single=np.sqrt(0.5 * ((uL**2).mean(-1) + (uR**2).mean(-1))),
                stages=stages, t_total=time.perf_counter() - t0)


def edges_points(p, I, doses, engine="auto", n_steps=30):
    """Края одной реализации для набора режимов, отличающихся только изображением и дозой (окно процесса).

    I — строки интенсивности (по одной на режим), doses — дозы. Все режимы разыгрывают одни и те же молекулы
    (общие случайные числа); в ядре C++ режимы считаются параллельно, по одному на поток.
    """
    I, doses = np.atleast_2d(I), np.asarray(doses, float)
    x = grid(p)
    ny = int(round(p.L / p.dy))
    pG = -np.expm1(-p.C_g * doses[:, None] * I)
    T, Tr = p.T_peb + 273.15, p.T_ref + 273.15
    D = arrhenius(p.D_ref, p.E_D, T, Tr)
    if engine == "cpp" or (engine == "auto" and _lercore is not None):
        kx, ky = diffusion_kernels(p, D, p.t_peb / n_steps)
        kx0, ky0 = gauss_kernels(p, p.r_c)
        q = dict(n_steps=n_steps, dx=p.dx, dy=p.dy, V=p.dx * p.dy * p.h_eff, c_pag=p.c_pag, c_q=p.c_q,
                 dt=p.t_peb / n_steps, kd=arrhenius(p.kd_ref, p.E_kd, T, Tr), t_dev=p.t_dev, R_min=p.R_min,
                 R_max=p.R_max, a_dev=p.a_dev, n_dev=p.n_dev, pitch=p.pitch)
        xL, xR, lost = _lercore.run(pG, kx, ky, kx0, ky0, ny, 1, p.seed, q, False)[:3]
        return xL, xR, lost.astype(bool)
    out = []
    for g in pG:
        rng = np.random.default_rng(p.seed)
        V = p.dx * p.dy * p.h_eff
        NG = rng.poisson(p.c_pag * V, (ny, x.size))
        cH = rng.binomial(NG, np.broadcast_to(g, NG.shape)) / V
        cQ = rng.poisson(p.c_q * V, NG.shape) / V
        out.append(develop(peb(cH, cQ, p, n_steps)[0], x, p))
    return tuple(np.array(v) for v in zip(*out))


def grid(p):
    nx = int(round(p.pitch / p.dx))
    nx += nx % 2
    return -p.pitch / 2 + (np.arange(nx) + 0.5) * p.dx


def mean_field(p, I=None):
    """Детерминированная цепочка на средних концентрациях: CD без шума, ~мс на точку."""
    x = grid(p)
    I = aerial_image(x, p.CD, p.pitch, p.wl, p.NA, p.sigma_s, p.focus) if I is None else I
    cH = (p.c_pag * -np.expm1(-p.C_g * p.dose * I))[None, :]
    d = peb(cH, np.full_like(cH, p.c_q), p)[0]
    l, r, lost = develop(d, x, p)
    return r[0] - l[0], bool(lost[0])
