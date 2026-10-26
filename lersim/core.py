import numpy as np

# Внутри ядра SI: координаты в м, PSD в м^3, f в циклах/м.


def freqs(N, dy):
    return np.fft.rfftfreq(N, dy)


def onesided_weight(N):
    a = np.full(N // 2 + 1, 2.0)
    a[0] = 1.0
    if N % 2 == 0:
        a[-1] = 1.0
    return a


def detrend(x, order=0):
    """G01: остаток u и база b после проекции на полином степени order вдоль последней оси."""
    x = np.asarray(x, float)
    N = x.shape[-1]
    t = np.linspace(-1.0, 1.0, N)
    V = np.vander(t, order + 1, increasing=True)
    Q, _ = np.linalg.qr(V)
    b = (x @ Q) @ Q.T
    return x - b, b


def edge_metrics(xL, xR, order=0):
    """G02–G04. Дисперсии с ddof=0, усреднение по ансамблю — по ведущим осям."""
    w = xR - xL
    uL, _ = detrend(xL, order)
    uR, _ = detrend(xR, order)
    vL, vR = (uL**2).mean(-1), (uR**2).mean(-1)
    C = (uL * uR).mean(-1)
    vw = ((uR - uL) ** 2).mean(-1)
    sL, sR, sw = np.sqrt(vL.mean()), np.sqrt(vR.mean()), np.sqrt(vw.mean())
    return dict(CD=w.mean(), sigma_L=sL, sigma_R=sR, sigma_w=sw,
                rho=C.mean() / (sL * sR) if sL * sR > 0 else np.nan,
                LER_L=3 * sL, LER_R=3 * sR, LWR=3 * sw)


def pair_spectra(uL, uR, dy):
    """G06–G07: односторонние S_LL, S_RR, комплексный S_LR, S_w, S_c (прямоугольное окно)."""
    N = uL.shape[-1]
    UL, UR = np.fft.rfft(uL), np.fft.rfft(uR)
    k = onesided_weight(N) * dy / N
    SLL, SRR, SLR = k * np.abs(UL) ** 2, k * np.abs(UR) ** 2, k * UL * np.conj(UR)
    return dict(f=freqs(N, dy), LL=SLL, RR=SRR, LR=SLR,
                w=SLL + SRR - 2 * SLR.real, c=0.25 * (SLL + SRR + 2 * SLR.real))


def band_rms(S, N, dy, band=None):
    """G08: сумма отсчётов, не трапеция — тогда Парсеваль выполняется точно."""
    df = 1.0 / (N * dy)
    return np.sqrt((S if band is None else S[..., band]).sum(-1) * df)


def psd_shape(f, sigma, xi, alpha, band=None):
    """G10: S_k = A[1+(2πfξ)^2]^-(α+1/2), A нормирован так, что Σ_B S_k Δf = σ_B^2.

    ξ здесь — масштабный параметр ℓ, совпадает с длиной спада 1/e только при α=1/2.
    """
    band = (f > 0) if band is None else band
    df = f[1] - f[0]
    g = (1 + (2 * np.pi * f * xi) ** 2) ** -(alpha + 0.5)
    S = np.where(band, g, 0.0)
    return S * sigma**2 / (S.sum() * df)


def synthesize_edges(N, dy, S, rho=0.0, n_lines=1, rng=None):
    """G11–G12: пара краёв с постоянной ρ, ожидание PSD каждого края = S.

    Реализации не перенормируются к точному σ — иначе меняется ансамбль.
    """
    rng = np.random.default_rng(rng)
    # Внутренние частоты: |U|^2 = N S/(4dy)·(a^2+b^2), т.е. E|U|^2 = N S/(2dy);
    # после irfft это даёт вклад S·Δf. На Найквисте коэффициент вещественный, E|U|^2 = N S/dy.
    amp = np.sqrt(N * S / (4 * dy))
    amp[0] = 0.0
    if N % 2 == 0:
        amp[-1] = np.sqrt(N * S[-1] / dy)
    def field():
        z = rng.standard_normal((n_lines, S.size)) + 1j * rng.standard_normal((n_lines, S.size))
        if N % 2 == 0:
            z[:, -1] = z[:, -1].real
        return np.fft.irfft(amp * z, n=N)
    g1, g2 = field(), field()
    return g1, rho * g1 + np.sqrt(1 - rho**2) * g2


def make_line(CD, L, dy, sigma, xi, alpha, rho=0.0, n_lines=1, rng=None):
    N = int(round(L / dy))
    f = freqs(N, dy)
    S = psd_shape(f, sigma, xi, alpha)
    uL, uR = synthesize_edges(N, dy, S, rho, n_lines, rng)
    y = np.arange(N) * dy
    return y, -CD / 2 + uL, CD / 2 + uR, f, S


def least_squares(fun, x0, iters=60):
    """Левенберг–Марквардт с численным якобианом для нескольких параметров (замена scipy.optimize).

    Пробные шаги с переполнением дают бесконечную невязку и просто отвергаются.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _lm(fun, np.asarray(x0, float), iters)


def _lm(fun, x, iters):
    r = fun(x)
    cost, lam = r @ r, 1e-3
    for _ in range(iters):
        h = 1e-6 * np.maximum(1.0, np.abs(x))
        J = np.stack([(fun(x + h[i] * np.eye(x.size)[i]) - r) / h[i] for i in range(x.size)], axis=1)
        A, g = J.T @ J, J.T @ r
        while True:
            dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            rn = fun(x + dx)
            cn = rn @ rn
            if np.isfinite(cn) and cn < cost:
                x, r, lam = x + dx, rn, max(lam / 3, 1e-12)
                break
            lam *= 4
            if lam > 1e12:
                return x
        if cost - cn < 1e-12 * max(cost, 1e-300):
            return x
        cost = cn
    return x


def fit_psd(f, S_est, band=None, p0=None, floor=False):
    """Подгонка G10 к ансамблевой периодограмме в лог-масштабе; возвращает σ_B, ξ, α, S_fit (и пол шума).

    floor=True добавляет белый пол S_n (некоррелированный шум края, M02) и уравнивает вклад декад:
    иначе высокочастотные отсчёты, которых большинство, определяют α.
    """
    band = (f > 0) if band is None else band
    fb, Sb = f[band], S_est[band]
    df = f[1] - f[0]
    w = np.ones_like(fb)
    if floor:
        k = np.floor(np.log10(fb) * 10)
        _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        w = 1 / np.sqrt(cnt[inv])
    x0 = list(np.log([Sb[:5].mean(), 10 / fb[-1], 1.0])) if p0 is None else list(np.log(p0))
    if floor:
        x0.append(np.log(Sb[-max(5, fb.size // 20):].mean() * 0.5))

    def model(p, ff):
        A, xi, a = np.exp(p[:3])
        S = A * (1 + (2 * np.pi * ff * xi) ** 2) ** -(a + 0.5)
        return S, S + (np.exp(p[3]) if floor else 0.0)

    p = least_squares(lambda p: w * (np.log(model(p, fb)[1]) - np.log(Sb)), x0)
    S_corr = model(p, f)[0]
    sigma = np.sqrt((S_corr * band).sum() * df)
    out = (sigma, np.exp(p[1]), np.exp(p[2]), model(p, f)[1])
    return out + (np.exp(p[3]),) if floor else out


def noise_correct_psd(S_obs, S_eps, Hm=1.0):
    """M02: отрицательные отсчёты сохраняются — их обнуление смещает интегральную оценку."""
    return (S_obs - S_eps) / np.abs(Hm) ** 2


def diffuse_periodic(c, dy, D, t):
    """R09: точное спектральное решение линейной диффузии на периодическом отрезке."""
    f = freqs(c.shape[-1], dy)
    return np.fft.irfft(np.fft.rfft(c) * np.exp(-D * (2 * np.pi * f) ** 2 * t), n=c.shape[-1])


def integrated_diffusion_transfer(f, D, t):
    """R10: H_eff = (1-e^{-a})/a, a = D(2πf)^2 t; предел при a→0 равен 1."""
    a = D * (2 * np.pi * f) ** 2 * t
    return np.where(a > 0, -np.expm1(-a) / np.where(a > 0, a, 1.0), 1.0)


def smooth_periodic(u, dy, nu2, nu4, t):
    """S04, детерминированный предел ζ=0."""
    k = 2 * np.pi * freqs(u.shape[-1], dy)
    return np.fft.irfft(np.fft.rfft(u) * np.exp(-(nu2 * k**2 + nu4 * k**4) * t), n=u.shape[-1])


def transfer_spectrum(T, S_in, Q=None, C=None):
    """S05 на каждой частоте: T,S_in,Q,C формы (n_f, 2, 2)."""
    Th = np.conj(np.swapaxes(T, -1, -2))
    S = T @ S_in @ Th
    if Q is not None:
        S = S + Q
    if C is not None:
        TC = T @ C
        S = S + TC + np.conj(np.swapaxes(TC, -1, -2))
    return S


kB = 8.617333262e-5  # эВ/К


def arrhenius(k_ref, Ea, T, T_ref):
    """R02 в относительной форме; T в кельвинах, Ea в эВ на частицу."""
    return k_ref * np.exp(-Ea / kB * (1 / T - 1 / T_ref))


def development_rate(d, R_min, R_max, a, n):
    """D01: нормированная форма Мака, R(0)=R_min, R(1)=R_max."""
    dn = np.clip(d, 0, 1) ** n
    return R_min + (R_max - R_min) * (a + 1) * dn / (a + dn)
