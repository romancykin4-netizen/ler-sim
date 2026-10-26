"""Пересоздаёт tests/data/golden.json — эталон детерминированных величин. Запускать только после
осознанного изменения модели; обычные правки кода эталон менять не должны."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from lersim.process import Process, aerial_image, grid, mean_field, simulate
from lersim.sweep import cd_matrix, latitude

p = replace(Process(), L=600.0)
x = grid(p)
pts = [(26.0, 0.0), (32.0, 0.0), (38.0, 0.0), (32.0, 100.0), (32.0, -150.0)]
doses, focuses = np.linspace(24, 42, 10), np.linspace(-300, 300, 13)
CD = cd_matrix(p, doses, focuses)
E0, EL, dof = latitude(CD, doses, focuses, p.CD)
gold = dict(
    image_x=x[::25].tolist(),
    image=aerial_image(x, p.CD, p.pitch, p.wl, p.NA, p.sigma_s, p.focus)[::25].tolist(),
    image_248=aerial_image(x, p.CD, p.pitch, 248.0, p.NA, p.sigma_s, p.focus)[::25].tolist(),
    mean_field_cd=[[d, f, float(mean_field(replace(p, dose=d, focus=f))[0])] for d, f in pts],
    nils=float(simulate(p, 1, "numpy")["NILS"]),
    dose_to_size_focus0=float(E0[focuses.size // 2]),
    exposure_latitude_focus0=float(EL[focuses.size // 2]),
    dof=float(dof),
)
Path(__file__).with_name("data").mkdir(exist_ok=True)
Path(__file__).with_name("data").joinpath("golden.json").write_text(json.dumps(gold, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in gold.items() if not isinstance(v, list)}, indent=1, ensure_ascii=False))
