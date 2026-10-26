import numpy as np
import matplotlib.pyplot as plt
from .style import apply, tag, NM, NM3


def summary_figure(r, p):
    apply()
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    a, b, c, d = ax.flat

    y = r["y"] * 1e6
    xL, xR = r["xL"][0] * NM, r["xR"][0] * NM
    a.fill_between(y, xL, xR, color="0.88", lw=0)
    a.plot(y, xL, "k", y, xR, "k")
    for s in (-1, 1):
        a.axhline(s * p["CD"] * NM / 2, color="0.35", ls="--", lw=0.6)
    a.set(xlim=(y[0], y[-1]), ylim=(-0.85 * p["CD"] * NM, 0.85 * p["CD"] * NM),
          xlabel=r"Position along line, $y$ ($\mu$m)", ylabel=r"Edge position, $x$ (nm)")
    m = r["m"]
    a.text(0.98, 0.5, f"CD = {m['CD']*NM:.1f} nm\nLER = {m['LER_L']*NM:.2f} nm (3$\\sigma$)\n"
           f"LWR = {m['LWR']*NM:.2f} nm (3$\\sigma$)", transform=a.transAxes,
           ha="right", va="center", fontsize=7, bbox=dict(fc="w", ec="0.6", lw=0.4, pad=2))
    tag(a, "a")

    f = r["f"][1:] / NM
    b.loglog(f, r["S_LL"][1:] * NM3, color="0.6", lw=0.6, label=f"Edge PSD, mean of {p['lines']} lines")
    b.loglog(f, r["S_w"][1:] * NM3, color="#1b3a6b", lw=0.6, alpha=0.8, label="Width PSD")
    b.loglog(f, r["S_in"][1:] * NM3, "k", lw=1.0, label="Input model (G10)")
    b.loglog(f, r["S_fit"][1:] * NM3, color="#8c1c13", ls="--", lw=0.9,
             label=rf"Fit: $\sigma$={r['fit'][0]*NM:.2f} nm, $\xi$={r['fit'][1]*NM:.1f} nm, $\alpha$={r['fit'][2]:.2f}")
    fk = 1 / (2 * np.pi * p["xi"]) / NM
    b.axvline(fk, color="0.3", ls=":", lw=0.7)
    b.text(fk * 1.12, b.get_ylim()[1] * 0.3, r"$1/2\pi\xi$", fontsize=7)
    b.set(xlim=(f[0], f[-1]), xlabel=r"Spatial frequency, $f$ (nm$^{-1}$)", ylabel=r"PSD (nm$^3$)")
    b.legend(loc="lower left")
    tag(b, "b")

    rho, ratio, se = r["rho_scan"]
    rr = np.linspace(-1, 1, 400)
    c.plot(rr, np.sqrt(2 * (1 - rr)), "k", label=r"$\sqrt{2(1-\rho)}$ (G04)")
    c.errorbar(rho, ratio, yerr=2 * se, fmt="o", ms=3, mfc="w", mec="#8c1c13", ecolor="#8c1c13",
               elinewidth=0.6, capsize=1.5, label=r"Monte Carlo, $\pm 2\,$SE")
    c.set(xlim=(-1.05, 1.05), ylim=(0, 2.1), xlabel=r"Edge correlation, $\rho_{LR}$", ylabel="LWR / LER")
    c.legend(loc="lower left")
    c.text(0.98, 0.97, "(c)", transform=c.transAxes, va="top", ha="right", fontweight="bold")

    s1 = r["sigma_single"] * NM
    d.hist(s1, bins=30, color="0.8", ec="0.4", lw=0.3, density=True)
    d.axvline(p["sigma"] * NM, color="k", lw=1.0, label=r"Ensemble $\sigma_B$")
    d.axvline(s1.mean(), color="#8c1c13", ls="--", lw=0.9, label="Mean of single lines")
    d.set(xlabel=r"Single-line RMS, $\sigma_i$ (nm)", ylabel="Probability density")
    d.text(0.98, 0.95, f"$L$ = {p['L']*1e6:.2f} $\\mu$m\nCV = {s1.std()/s1.mean()*100:.1f}%",
           transform=d.transAxes, ha="right", va="top", fontsize=7)
    d.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))
    tag(d, "d")
    return fig
