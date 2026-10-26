import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lersim.core import make_line, edge_metrics, detrend, pair_spectra, fit_psd, band_rms
from lersim.plots import summary_figure

nm = 1e-9

ap = argparse.ArgumentParser(description="LER/LWR: синтез краёв по G10–G12 и спектральная проверка")
ap.add_argument("--CD", type=float, default=120.0, help="нм")
ap.add_argument("--L", type=float, default=2048.0, help="нм")
ap.add_argument("--dx", type=float, default=1.0, help="нм")
ap.add_argument("--sigma", type=float, default=2.25, help="нм, RMS одного края в полосе")
ap.add_argument("--xi", type=float, default=20.0, help="нм")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--lines", type=int, default=500)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="out")
a = ap.parse_args()

p = dict(CD=a.CD * nm, L=a.L * nm, dx=a.dx * nm, sigma=a.sigma * nm, xi=a.xi * nm,
         alpha=a.alpha, rho=a.rho, lines=a.lines, seed=a.seed)
rng = np.random.default_rng(a.seed)

y, xL, xR, f, S_in = make_line(p["CD"], p["L"], p["dx"], p["sigma"], p["xi"], p["alpha"],
                               p["rho"], p["lines"], rng)
N = y.size
m = edge_metrics(xL, xR)
uL, uR = detrend(xL)[0], detrend(xR)[0]
sp = pair_spectra(uL, uR, p["dx"])
S_LL, S_w = sp["LL"].mean(0), sp["w"].mean(0)
fit = fit_psd(f, S_LL)

rho_grid = np.linspace(-0.9, 0.9, 7)
ratio, se = [], []
for rho in rho_grid:
    _, l, r, *_ = make_line(p["CD"], p["L"], p["dx"], p["sigma"], p["xi"], p["alpha"], rho, 200, rng)
    ul, ur = detrend(l)[0], detrend(r)[0]
    q = np.sqrt(((ur - ul) ** 2).mean(-1) / (0.5 * (ul**2 + ur**2).mean(-1)))
    ratio.append(q.mean()); se.append(q.std(ddof=1) / np.sqrt(q.size))

res = dict(y=y, xL=xL, xR=xR, f=f, S_in=S_in, S_LL=S_LL, S_w=S_w, S_fit=fit[3], fit=fit[:3], m=m,
           sigma_single=np.sqrt((uL**2).mean(-1)), rho_scan=(rho_grid, np.array(ratio), np.array(se)))

out = Path(a.out); out.mkdir(exist_ok=True)
fig = summary_figure(res, p)
fig.savefig(out / "ler_summary.pdf"); fig.savefig(out / "ler_summary.png", dpi=300)
plt.close(fig)

# Ожидание σ^2 одной линии после удаления среднего равно σ_B^2 только потому, что DC исключён из полосы.
sigma_parseval = band_rms(S_LL, N, p["dx"])
report = dict(params={k: v / nm if k in ("CD", "L", "dx", "sigma", "xi") else v for k, v in p.items()},
              N=N, CD_nm=m["CD"] / nm, sigma_L_nm=m["sigma_L"] / nm, sigma_R_nm=m["sigma_R"] / nm,
              LER_nm=m["LER_L"] / nm, LWR_nm=m["LWR"] / nm, rho_est=m["rho"],
              sigma_from_psd_nm=sigma_parseval / nm,
              fit=dict(sigma_nm=round(float(fit[0] / nm), 4), xi_nm=round(float(fit[1] / nm), 3), alpha=round(float(fit[2]), 4)),
              single_line_sigma_cv=float(res["sigma_single"].std() / res["sigma_single"].mean()))
(out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
np.savetxt(out / "edges_line0.csv", np.c_[y / nm, xL[0] / nm, xR[0] / nm], delimiter=",",
           header="y_nm,xL_nm,xR_nm", comments="", fmt="%.6f")
np.savez_compressed(out / "ensemble.npz", y=y, xL=xL, xR=xR, f=f, S_in=S_in, S_LL=S_LL, S_w=S_w)

w = max(map(len, report))
for k, v in report.items():
    if k != "params":
        print(f"{k:<{w}}  {v if not isinstance(v, float) else f'{v:.4g}'}")
