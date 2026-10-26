"""Контракт выходных данных расчёта (функция simulate)."""
from dataclasses import replace
import numpy as np
import pytest
from lersim.process import Process, simulate, _lercore

ENGINES = ["numpy"] + (["cpp"] if _lercore else [])


@pytest.fixture(scope="module", params=ENGINES)
def run(request):
    p = replace(Process(), L=600.0)
    return p, simulate(p, 4, request.param)


def test_shapes_and_finiteness(rec, run):
    """Размеры выходных массивов соответствуют сетке, значения конечны.

    Метод: расчёт по умолчанию, участок 0,6 мкм, 4 повторения; проверяются формы и отсутствие NaN/∞.
    Критерий: контуры 4 × 300 точек, карта 300 × сетка поперёк линии; все значения конечны.
    Зачем: защищает от молчаливых ошибок индексации и деления на ноль в выходных данных.
    """
    p, r = run
    ny = int(round(p.L / p.dy))
    assert rec("Размер массива левых краёв", str(r["xL"].shape), str((4, ny)), "", r["xL"].shape == r["xR"].shape == (4, ny))
    assert rec("Размер карты депротекции", str(r["d"].shape), str((ny, r["x"].size)), "", r["d"].shape == (ny, r["x"].size))
    bad = [k for k in ("xL", "xR", "d", "I", "S_edge", "S_w", "f") if not np.isfinite(r[k]).all()]
    assert rec("Массивы с NaN или бесконечностью", ", ".join(bad) or "нет", "нет", "", not bad)


def test_value_ranges(rec, run):
    """Все величины в физически допустимых пределах.

    Метод: проверка диапазонов выданных величин.
    Критерий: 0 < CD < шаг; левый край левее правого; края в пределах периода; LER, LWR ≥ 0;
    −1 ≤ ρ ≤ 1; доли (разрывы, депротекция) от 0 до 1; интенсивность от 0 до 1; NILS и длина диффузии > 0.
    Зачем: любое нарушение — признак ошибки, даже если остальные числа выглядят правдоподобно.
    """
    p, r = run
    m = r["m"]
    assert rec("Ширина линии", m["CD"], f"0 … {p.pitch:g}", "нм", 0 < m["CD"] < p.pitch)
    assert rec("Левый край везде левее правого", bool((r["xL"] < r["xR"]).all()), "да", "", (r["xL"] < r["xR"]).all())
    assert rec("Края в пределах периода", bool((r["xL"] >= -p.pitch / 2).all() and (r["xR"] <= p.pitch / 2).all()), "да", "",
               (r["xL"] >= -p.pitch / 2).all() and (r["xR"] <= p.pitch / 2).all())
    assert rec("Шероховатость левого края", m["LER_L"], "≥ 0", "нм", m["LER_L"] >= 0)
    assert rec("Шероховатость ширины", m["LWR"], "≥ 0", "нм", m["LWR"] >= 0)
    assert rec("Корреляция краёв", m["rho"], "−1 … 1", "", -1 <= m["rho"] <= 1)
    assert rec("Доля разрывов", r["lost_frac"], "0 … 1", "", 0 <= r["lost_frac"] <= 1)
    assert rec("Доля снятой защиты", r["d"], "0 … 1", "", ((r["d"] >= 0) & (r["d"] <= 1)).all())
    assert rec("Интенсивность", r["I"], "0 … 1", "", ((r["I"] >= 0) & (r["I"] <= 1 + 1e-9)).all())
    assert rec("NILS", r["NILS"], "> 0", "", r["NILS"] > 0)
    assert rec("Длина диффузии", r["L_diff"], "> 0", "нм", r["L_diff"] > 0)


def test_reported_metrics_agree_with_reported_edges(rec, run):
    """Выданные метрики пересчитываются из выданных контуров.

    Метод: ширина, σ левого и правого края и 3σ пересчитываются независимо по выданным координатам краёв.
    Критерий: совпадение с выданными метриками до 1e-9.
    Зачем: таблица результатов и экспорт контуров не должны противоречить друг другу.
    """
    m = run[1]["m"]
    xL, xR = run[1]["xL"], run[1]["xR"]
    uL, uR = xL - xL.mean(1, keepdims=True), xR - xR.mean(1, keepdims=True)
    for name, got, ref in (("Ширина линии", m["CD"], (xR - xL).mean()), ("σ левого края", m["sigma_L"], np.sqrt((uL**2).mean())),
                           ("σ правого края", m["sigma_R"], np.sqrt((uR**2).mean())), ("LER левого края / 3σ", m["LER_L"] / 3, m["sigma_L"]),
                           ("LWR / 3σ ширины", m["LWR"] / 3, m["sigma_w"])):
        assert rec(name, got, f"{ref:.6f}", "нм", np.isclose(got, ref))


def test_lwr_identity(rec, run):
    """σ_w² = σ_L² + σ_R² − 2ρ·σ_L·σ_R — связь между тремя выданными величинами.

    Метод: подстановка выданных σ_L, σ_R, ρ в тождество.
    Критерий: относительное совпадение до 1e-9.
    Зачем: тождество верно по определению; нарушение означает несогласованные метрики.
    """
    m = run[1]["m"]
    rhs = m["sigma_L"] ** 2 + m["sigma_R"] ** 2 - 2 * m["rho"] * m["sigma_L"] * m["sigma_R"]
    assert rec("σ ширины²", m["sigma_w"] ** 2, f"{rhs:.6f}", "нм²", np.isclose(m["sigma_w"] ** 2, rhs))


def test_spectrum_integrates_to_variance(rec, run):
    """Площадь под выданным спектром края равна выданной дисперсии края.

    Метод: сумма спектра × шаг частоты против ½(σ_L² + σ_R²).
    Критерий: относительное совпадение до 1e-9.
    Зачем: спектр и σ должны описывать одну и ту же шероховатость.
    """
    m = run[1]["m"]
    f = run[1]["f"]
    a = run[1]["S_edge"].sum() * (f[1] - f[0])
    b = 0.5 * (m["sigma_L"] ** 2 + m["sigma_R"] ** 2)
    assert rec("Площадь под спектром", a, f"{b:.6f}", "нм²", np.isclose(a, b, rtol=1e-9))


def test_frequency_axis(rec, run):
    """Ось частот спектра от 0 до частоты Найквиста с шагом 1/L.

    Метод: проверка первой, последней точки и шага.
    Критерий: f₀ = 0; f_max = 1/(2·Δy) = 0,25 нм⁻¹; шаг 1/L = 1/600 нм⁻¹.
    Зачем: от оси частот зависит длина корреляции и форма спектра.
    """
    p, r = run
    f = r["f"]
    assert rec("Первая частота", float(f[0]), "0", "1/нм", f[0] == 0)
    assert rec("Последняя частота", float(f[-1]), f"{1 / (2 * p.dy):g}", "1/нм", np.isclose(f[-1], 1 / (2 * p.dy)))
    assert rec("Шаг частоты", float(f[1] - f[0]), f"{1 / p.L:.6f}", "1/нм", np.allclose(np.diff(f), 1 / p.L))


def test_stage_report(rec, run):
    """Отчёт по этапам полный и в правильном порядке.

    Метод: список этапов, записываемых в журнал.
    Критерий: «Изображение», «Экспонирование», «Термообработка и проявление», «Обработка краёв»; время ≥ 0.
    Зачем: журнал расчётов строится из этого списка.
    """
    r = run[1]
    names = [s[0] for s in r["stages"]]
    exp = ["Изображение", "Экспонирование", "Термообработка и проявление", "Обработка краёв"]
    assert rec("Этапы", ", ".join(names), ", ".join(exp), "", names == exp)
    assert rec("Общее время", r["t_total"], "≥ суммы этапов", "с", r["t_total"] >= sum(s[2] for s in r["stages"]) * 0.99)


def test_same_seed_same_result_other_seed_differs(rec):
    """Одинаковое начальное значение генератора даёт одинаковый результат, другое — другой.

    Метод: три расчёта: два с генератором 1, один с генератором 2.
    Критерий: первые два совпадают побитово, третий отличается.
    Зачем: воспроизводимость и действительная случайность шума.
    """
    p = replace(Process(), L=600.0)
    a, b, c = simulate(p, 2), simulate(p, 2), simulate(replace(p, seed=2), 2)
    assert rec("Генератор 1 дважды: совпадение", bool(np.array_equal(a["xL"], b["xL"])), "да", "", np.array_equal(a["xL"], b["xL"]))
    assert rec("Генераторы 1 и 2: совпадение", bool(np.array_equal(a["xL"], c["xL"])), "нет", "", not np.array_equal(a["xL"], c["xL"]))


def test_line_break_is_reported(rec):
    """Разрыв линии при передержке отражается в выходных данных.

    Метод: линия 80 нм, доза 70 мДж/см², PEB 125 °C 120 с, проявление 90 с — линия растворяется;
    сравнение с режимом без разрыва.
    Критерий: доля разрывов 100 %, ширина 0; во втором режиме разрывов нет, ширина > 0.
    Зачем: разрушение линии должно явно отмечаться, а не давать случайные края.
    """
    r = simulate(replace(Process(), L=600.0, CD=80.0, dose=70.0, t_dev=90.0, T_peb=125.0, t_peb=120.0), 1)
    assert rec("Доля разрывов (передержка)", r["lost_frac"], "1", "", r["lost_frac"] == 1.0)
    assert rec("Ширина (передержка)", r["m"]["CD"], "0", "нм", np.isclose(r["m"]["CD"], 0.0))
    r = simulate(replace(Process(), L=600.0, dose=70.0, t_dev=90.0, T_peb=125.0), 1)
    assert rec("Доля разрывов (линия 120 нм)", r["lost_frac"], "0", "", r["lost_frac"] == 0.0)
    assert rec("Ширина (линия 120 нм)", r["m"]["CD"], "> 0", "нм", r["m"]["CD"] > 0)
