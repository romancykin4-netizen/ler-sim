import numpy as np
import matplotlib as mpl
from .style import RC

RED, BLUE = "#8c1c13", "#1b3a6b"

SCREEN = dict(RC, **{"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "lines.linewidth": 1.0, "axes.titlesize": 9})


DEP = mpl.colors.LinearSegmentedColormap.from_list("dep", ["white", "0.4"])

# Подписи по нормам научных публикаций: «Величина символ, единица»; переменные — курсивом
# ($x$, $f$, $E$), аббревиатуры (CD, LER, NILS) — прямым шрифтом; десятичная запятая на шкалах.
L_Y = "Координата вдоль линии $y$, мкм"
L_X = "Координата поперёк линии $x$, нм"
L_F = "Пространственная частота $f$, нм$^{-1}$"
L_PSD = "Спектральная плотность мощности $S(f)$, нм$^{3}$"
L_DZ = "Расфокусировка $\\Delta z$, нм"
L_E = "Доза экспонирования $E$, мДж/см$^{2}$"
L_CD = "Ширина линии CD, нм"
L_LER = "Шероховатость края LER (3$\\sigma$), нм"


class Comma(mpl.ticker.ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def _comma(*axes):
    for ax in axes:
        for a in (ax.xaxis, ax.yaxis):
            if a.get_scale() == "linear":
                a.set_major_formatter(Comma())


def _panel(ax, s):
    ax.set_title(s, loc="left", fontsize=10)


def _ax(fig, n=1):
    mpl.rcParams.update(SCREEN)
    fig.clear()
    fig._live = None
    fig.set_layout_engine("constrained")
    return fig.subplots(1, n) if n > 1 else fig.add_subplot()


def _empty(fig, text):
    mpl.rcParams.update(SCREEN)
    fig.clear()
    fig._live = None
    fig.text(0.5, 0.5, text, ha="center", va="center", color="0.35", fontsize=10)


def _live(fig, kind, sig):
    """Ручки элементов графика, если его можно обновить на месте (та же структура данных)."""
    h = getattr(fig, "_live", None)
    if h is None or h["kind"] != kind or h["sig"] != sig:
        return None
    # Раскладка уже посчитана при первой отрисовке; пересчитывать её на каждом кадре дорого.
    if not getattr(fig, "_relayout", False):
        fig.set_layout_engine("none")
    return h


def _gray(d):
    """Карта депротекции сразу в 8-битных оттенках (как DEP): рендеру не нужна цветовая карта и прозрачность."""
    k = max(1, d.shape[0] // 400)
    g = (255 * (1 - 0.6 * np.clip(d[::k].T, 0, 1))).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def logbin(f, S, n=40):
    e = np.geomspace(f[1], f[-1], n + 1)
    i = np.digitize(f[1:], e)
    k = [j for j in range(1, n + 1) if (i == j).any()]
    return (np.array([np.exp(np.log(f[1:][i == j]).mean()) for j in k]),
            np.array([S[1:][i == j].mean() for j in k]))


def contour(fig, r, p, ref=None):
    y = r["y"] / 1e3
    ext = (y[0], y[-1], r["x"][0], r["x"][-1])
    h = _live(fig, "contour", (r["d"].shape, ref is not None))
    if h:
        h["im"].set_data(_gray(r["d"]))
        h["im"].set_extent(ext)
        h["L"].set_data(y, r["xL"][0])
        h["R"].set_data(y, r["xR"][0])
        h["t1"].set_ydata([p.CD / 2] * 2)
        h["t2"].set_ydata([-p.CD / 2] * 2)
        if ref is not None:
            yr = ref["y"] / 1e3
            h["rL"].set_data(yr, ref["xL"][0])
            h["rR"].set_data(yr, ref["xR"][0])
        h["ax"].set(xlim=ext[:2], ylim=ext[2:])
        return
    ax = _ax(fig)
    h = dict(kind="contour", sig=(r["d"].shape, ref is not None), ax=ax)
    h["im"] = ax.imshow(_gray(r["d"]), origin="lower", aspect="auto", extent=ext, interpolation="nearest")
    h["L"], = ax.plot(y, r["xL"][0], "k", lw=0.9, label="Край резиста после проявления")
    h["R"], = ax.plot(y, r["xR"][0], "k", lw=0.9)
    h["t1"] = ax.axhline(p.CD / 2, color="0.3", ls="--", lw=0.7, label="Номинальные границы линии")
    h["t2"] = ax.axhline(-p.CD / 2, color="0.3", ls="--", lw=0.7)
    if ref is not None:
        yr = ref["y"] / 1e3
        h["rL"], = ax.plot(yr, ref["xL"][0], color=RED, lw=0.7, label="Опорный режим")
        h["rR"], = ax.plot(yr, ref["xR"][0], color=RED, lw=0.7)
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.set(xlabel=L_Y, ylabel=L_X, xlim=ext[:2], ylim=ext[2:])
    ax.grid(False)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 1), DEP), ax=ax, pad=0.01, aspect=30)
    cb.set_label("Степень депротекции $d$")
    cb.ax.minorticks_off()
    _comma(ax, cb.ax)
    fig._live = h


def section(fig, r, p):
    x = r["x"]
    a0 = r["acid0"] / r["acid0"].max()
    eL, eR = r["xL"].mean(), r["xR"].mean()
    h = _live(fig, "section", x.size)
    if h:
        h["I"].set_data(x, r["I"])
        h["a"].set_data(x, a0)
        h["d"].set_data(x, r["d_mean"])
        h["eL"].set_xdata([eL] * 2)
        h["eR"].set_xdata([eR] * 2)
        h["tL"].set_xdata([-p.CD / 2] * 2)
        h["tR"].set_xdata([p.CD / 2] * 2)
        h["ax"].set_xlim(x[0], x[-1])
        return
    ax = _ax(fig)
    h = dict(kind="section", sig=x.size, ax=ax)
    h["I"], = ax.plot(x, r["I"], "k", label="Интенсивность $I/I_0$")
    h["a"], = ax.plot(x, a0, color="0.6", lw=0.7, label="Концентрация кислоты после экспонирования (норм.)")
    h["d"], = ax.plot(x, r["d_mean"], color=BLUE, ls="--", label="Степень депротекции $d$ после PEB")
    h["eL"] = ax.axvline(eL, color=RED, lw=0.9, label="Средние положения краёв")
    h["eR"] = ax.axvline(eR, color=RED, lw=0.9)
    h["tL"] = ax.axvline(-p.CD / 2, color="0.3", ls=":", lw=0.9, label="Номинальные положения краёв")
    h["tR"] = ax.axvline(p.CD / 2, color="0.3", ls=":", lw=0.9)
    ax.set(xlim=(x[0], x[-1]), ylim=(0, 1.05), xlabel=L_X, ylabel="Нормированная величина")
    _comma(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3, frameon=False)
    fig._live = h


def psd(fig, r, fit):
    f, S = r["f"], r["S_edge"]
    fb, Sb = logbin(f, S)
    fw, Sw = logbin(f, r["S_w"])
    h = _live(fig, "psd", (f.size, len(fit) > 4))
    if h:
        h["raw"].set_data(f[1:], S[1:])
        h["e"].set_data(fb, Sb)
        h["w"].set_data(fw, Sw)
        h["fit"].set_data(f[1:], fit[3][1:])
        if len(fit) > 4:
            h["n"].set_ydata([fit[4]] * 2)
        ax = h["ax"]
        ax.relim()
        ax.autoscale_view(scalex=False)
        return
    ax = _ax(fig)
    h = dict(kind="psd", sig=(f.size, len(fit) > 4), ax=ax)
    h["raw"], = ax.loglog(f[1:], S[1:], color="0.82", lw=0.5)
    h["e"], = ax.loglog(fb, Sb, "o", ms=3, mfc="w", mec="k", label="Край линии, $S_{\\mathrm{LER}}$")
    h["w"], = ax.loglog(fw, Sw, "s", ms=3, mfc="w", mec=BLUE, label="Ширина линии, $S_{\\mathrm{LWR}}$")
    h["fit"], = ax.loglog(f[1:], fit[3][1:], color=RED, lw=1.1, label="Аппроксимация моделью Паласанцаса")
    if len(fit) > 4:
        h["n"] = ax.axhline(fit[4], color=RED, ls=":", lw=0.8, label="Уровень белого шума $S_0$")
    ax.set(xlim=(f[1], f[-1]), xlabel=L_F, ylabel=L_PSD)
    ax.legend(loc="lower left")
    fig._live = h


def ensemble(fig, r):
    s = r["sigma_single"]
    n = s.size
    if n < 2:
        _empty(fig, "Для построения задайте число реализаций N > 1\n(вкладка «Расчёт и резист», группа «Параметры численного расчёта»)")
        return
    a, b = _ax(fig, 2)
    a.hist(3 * s, bins=min(30, max(6, n // 2)), color="0.85", ec="0.35", lw=0.4)
    a.axvline(3 * np.sqrt((s**2).mean()), color="k", lw=1.0, label="Оценка по ансамблю")
    a.set(xlabel="Шероховатость края в реализации LER$_i$ (3$\\sigma$), нм", ylabel="Число реализаций")
    a.legend(loc="upper right")
    _panel(a, "а)")
    k = np.arange(1, n + 1)
    run = 3 * np.sqrt(np.cumsum(s**2) / k)
    se = np.array([np.std(3 * s[:i], ddof=1) / np.sqrt(i) if i > 1 else np.nan for i in k])
    b.fill_between(k, run - 2 * se, run + 2 * se, color="0.88", lw=0, label="Доверительный интервал 95 %")
    b.plot(k, run, "k", marker="o", ms=2.5, label="Оценка по $N$ реализациям")
    b.set(xlabel="Число реализаций $N$", ylabel=L_LER, xlim=(1, n))
    b.legend(loc="upper right")
    _panel(b, "б)")
    _comma(a, b)


def bossung(fig, pw, p):
    ax = _ax(fig)
    D, F, CD = pw["doses"], pw["focuses"], pw["CD"]
    ax.axhspan(0.9 * p.CD, 1.1 * p.CD, color="0.92", lw=0, label="Допуск $\\pm$10 %")
    ax.axhline(p.CD, color="0.5", lw=0.6, ls="--")
    cm = mpl.colormaps["Greys"]
    mk = "osd^v<>ph*"
    for i, d in enumerate(D):
        ax.plot(F, CD[i], marker=mk[i % len(mk)], ms=3, mfc="w", lw=0.8,
                color=cm(0.35 + 0.65 * i / max(1, D.size - 1)), label=f"{d:.1f}".replace(".", ","))
    ax.set(xlabel=L_DZ, ylabel=L_CD, xlim=(F[0], F[-1]))
    ax.legend(title="$E$, мДж/см$^{2}$", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, title_fontsize=8)
    _comma(ax)


def window(fig, pw, p):
    a, b = _ax(fig, 2)
    D, F, CD = pw["doses"], pw["focuses"], pw["CD"]
    cs = a.contourf(F, D, CD, levels=14, cmap="Greys_r", alpha=0.85)
    a.contour(F, D, CD, levels=[0.9 * p.CD, 1.1 * p.CD], colors="k", linewidths=1.0)
    a.contour(F, D, CD, levels=[p.CD], colors=RED, linewidths=1.2)
    if pw.get("LER") is not None and np.isfinite(pw["LER"]).any():
        c = a.contour(F, D, pw["LER"], levels=6, colors=BLUE, linewidths=0.7, linestyles="--")
        a.clabel(c, fmt=lambda v: f"{v:.1f}".replace(".", ","), fontsize=7)
    a.set(xlabel=L_DZ, ylabel=L_E)
    a.grid(False)
    _panel(a, "а)")
    cb = fig.colorbar(cs, ax=a, pad=0.01, aspect=30)
    cb.set_label(L_CD)
    cb.ax.minorticks_off()
    b.plot(F, pw["EL"], "k", marker="o", ms=3, mfc="w", label="EL при допуске $\\pm$10 % по CD")
    b.axhline(5, color="0.5", ls=":", lw=0.8, label="Порог EL = 5 %")
    b.set(xlabel=L_DZ, ylabel="Энергетическая широта EL, %", xlim=(F[0], F[-1]), ylim=(0, None))
    b.legend(loc="lower center")
    _panel(b, "б)")
    b2 = b.twinx()
    b2.plot(F, pw["E0"], color=RED, lw=1.0)
    b2.set_ylabel("Доза номинального размера $E_0$, мДж/см$^{2}$", color=RED)
    b2.tick_params(axis="y", colors=RED, which="both")
    b2.grid(False)
    _comma(a, b, b2, cb.ax)


def _band(ax, k, q, label_ci, label_pi):
    s = q["total"]
    ax.axhspan(*s["pi"], color="0.9", lw=0, label=label_pi)
    ax.axhspan(*s["ci"], color="0.72", lw=0, label=label_ci)
    ax.axhline(s["mean"], color="k", lw=1.0, label="Среднее $\\bar{x}$")
    ax.plot(k, q["values"], "o", ms=3, mfc="w", mec=BLUE, mew=0.8, label="Отдельные реализации")


def stats(fig, est):
    """а) распределение отклонений точек края от номинала; б), в) CD и LER по реализациям с интервалами."""
    a, b, c = _ax(fig, 3)
    ps, pts = est["point_stats"], est["points"]
    pc = f"{100 * est['conf']:g}".replace(".", ",")
    m, s = ps["mean"], ps["sd"]
    a.hist(pts, bins=60, density=True, color="0.85", ec="0.45", lw=0.3, label="Расчёт (все точки краёв)")
    x = np.linspace(pts.min(), pts.max(), 300)
    a.plot(x, np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)), color=RED, lw=1.1,
           label="Нормальное распределение")
    a.axvspan(*ps["band"], color=BLUE, alpha=0.08, lw=0)
    for v in ps["band"]:
        a.axvline(v, color=BLUE, ls="--", lw=0.8)
    a.axvline(m, color="k", lw=1.0)
    a.plot([], [], color=BLUE, ls="--", lw=0.8, label=f"$\\bar{{\\delta}} \\pm z\\sigma$ ({pc} %)")
    a.set(xlabel="Отклонение края от номинала $\\delta$, нм", ylabel="Плотность вероятности, нм$^{-1}$")
    lim = max(abs(pts.min() - m), abs(pts.max() - m))
    a.set_xlim(m - lim, m + lim)
    a.set_ylim(0, 1.1 / (s * np.sqrt(2 * np.pi)))
    _panel(a, "а)")
    k = np.arange(1, est["n"] + 1)
    for ax, key, lab in ((b, "cd", L_CD), (c, "ler", "Шероховатость LER (3$\\sigma$), нм")):
        _band(ax, k, est["quantities"][key], f"ДИ среднего {pc} %", f"Интервал предсказания {pc} %")
        ax.set(xlabel="Номер реализации $i$", ylabel=lab, xlim=(0.5, est["n"] + 0.5))
    b.axhline(est["p"].CD, color="0.4", ls=":", lw=0.9, label="Номинал CD$_0$")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncols=4, frameon=False)
    _panel(b, "б)")
    _panel(c, "в)")
    _comma(a, b, c)
