"""Детерминированные результаты в сравнении с эталонными значениями (tests/data/golden.json).

Выявляет непреднамеренные изменения модели. Эталон пересоздаётся только осознанно: python -m tests.make_golden"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from lersim.process import Process, aerial_image, grid, mean_field, simulate
from lersim.sweep import cd_matrix, latitude

G = json.loads(Path(__file__).resolve().parents[1].joinpath("data", "golden.json").read_text(encoding="utf-8"))
p = replace(Process(), L=600.0)


def test_aerial_image(rec):
    """Воздушное изображение совпадает с эталоном.

    Метод: изображение по умолчанию при 193 и 248 нм в 13 точках поперёк периода против сохранённых значений.
    Критерий: относительное отклонение не больше 1e-9.
    Зачем: любое изменение оптического расчёта, намеренное или нет, будет замечено.
    """
    x = grid(p)
    assert rec("Узлы сетки совпадают с эталоном", bool(np.allclose(x[::25], G["image_x"])), "да", "", np.allclose(x[::25], G["image_x"]))
    for wl, key in ((p.wl, "image"), (248.0, "image_248")):
        I = aerial_image(x, p.CD, p.pitch, wl, p.NA, p.sigma_s, p.focus)[::25]
        e = float(np.abs(I / np.array(G[key]) - 1).max())
        rec(f"Интенсивность в центре линии, {wl:g} нм", float(I[len(I) // 2]), f"{G[key][len(I) // 2]:.6f}")
        assert rec(f"Отклонение от эталона, {wl:g} нм", e, "≤ 1e-9", "", e <= 1e-9)


def test_mean_field_cd(rec):
    """Ширина линии по средним концентрациям совпадает с эталоном.

    Метод: пять режимов доза/фокус: (26, 0), (32, 0), (38, 0), (32, 100), (32, −150).
    Критерий: относительное отклонение не больше 1e-7.
    Зачем: фиксирует химию, диффузию и проявление в детерминированном режиме.
    """
    for d, f, cd in G["mean_field_cd"]:
        v = mean_field(replace(p, dose=d, focus=f))[0]
        assert rec(f"Ширина при дозе {d:g} мДж/см², фокусе {f:+g} нм", v, f"{cd:.4f}", "нм", np.isclose(v, cd, rtol=1e-7))


def test_nils(rec):
    """NILS на заданном крае совпадает с эталоном.

    Метод: NILS из расчёта по умолчанию.
    Критерий: относительное отклонение не больше 1e-9.
    Зачем: NILS — ключевая характеристика изображения в таблице результатов.
    """
    v = simulate(p, 1)["NILS"]
    assert rec("NILS", v, f"{G['nils']:.6f}", "", np.isclose(v, G["nils"], rtol=1e-9))


def test_process_window(rec):
    """Окно процесса совпадает с эталоном.

    Метод: сетка 10 доз × 13 фокусов, допуск ±10 %, порог энергетической широты 5 %.
    Критерий: доза номинального размера и энергетическая широта EL при фокусе 0 — до 1e-6; глубина фокуса — точно.
    Зачем: фиксирует расчёт окна процесса от изменений.
    """
    doses, focuses = np.linspace(24, 42, 10), np.linspace(-300, 300, 13)
    E0, EL, dof = latitude(cd_matrix(p, doses, focuses), doses, focuses, p.CD)
    j = focuses.size // 2
    assert rec("Доза номинального размера при фокусе 0", E0[j], f"{G['dose_to_size_focus0']:.4f}", "мДж/см²",
               np.isclose(E0[j], G["dose_to_size_focus0"], rtol=1e-7))
    assert rec("Энергетическая широта EL при фокусе 0", EL[j], f"{G['exposure_latitude_focus0']:.4f}", "%",
               np.isclose(EL[j], G["exposure_latitude_focus0"], rtol=1e-6))
    assert rec("Глубина фокуса DOF", dof, f"{G['dof']:g}", "нм", dof == G["dof"])
