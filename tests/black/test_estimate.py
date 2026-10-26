"""Статистическая оценка результата с интервалами погрешности (функция estimate)."""
from dataclasses import replace
import numpy as np
import pytest
from lersim.process import Process, simulate
from lersim.stats import estimate, per_line

P = replace(Process(), L=600.0)


@pytest.fixture(scope="module")
def plain():
    return estimate(P, 16, 0.95)


@pytest.fixture(scope="module")
def varied():
    return estimate(P, 16, 0.95, dict(dose=3.0, focus=40.0))


def test_interval_structure(rec, plain):
    """Интервалы упорядочены: среднее внутри ДИ, ДИ внутри интервала предсказания.

    Метод: оценка без нестабильности процесса, N = 16, P = 0,95.
    Критерий: для δ, CD, LER, LWR: нижняя граница интервала предсказания ≤ ДИ ≤ x̄ ≤ ДИ ≤ верхняя граница;
    вклад нестабильности процесса 0; число значений равно N.
    Зачем: защищает от перепутанных границ и неверного числа реализаций в таблице.
    """
    for key, q in plain["quantities"].items():
        s = q["total"]
        ok = s["pi"][0] <= s["ci"][0] <= s["mean"] <= s["ci"][1] <= s["pi"][1]
        assert rec(f"{q['name']}: порядок границ", ok, "да", "", ok)
        assert rec(f"{q['name']}: число значений", q["values"].size, "16", "", q["values"].size == 16)
        assert rec(f"{q['name']}: вклад нестабильности", q["process_share"], "0", "", q["process_share"] == 0)


def test_mean_agrees_with_ensemble_run(rec, plain):
    """Средняя ширина и шероховатость оценки согласуются с обычным расчётом по ансамблю.

    Метод: обычный расчёт 16 реализаций с другим генератором против оценки.
    Критерий: средняя CD и LER ансамбля внутри интервала предсказания оценки; разность средних CD меньше
    4 стандартных ошибок разности.
    Зачем: оценка и основной расчёт описывают одну и ту же линию и не должны противоречить друг другу.
    """
    r = simulate(replace(P, seed=777), 16, keep_map=False)
    q = per_line(r, P)
    cd = plain["quantities"]["cd"]["total"]
    se = np.hypot(cd["sd"] / 4, q["cd"].std(ddof=1) / 4)
    assert rec("Разность средних CD", abs(q["cd"].mean() - cd["mean"]), f"≤ {4 * se:.3f}", "нм", abs(q["cd"].mean() - cd["mean"]) <= 4 * se)
    lr = plain["quantities"]["ler"]["total"]
    assert rec("LER ансамбля в интервале предсказания", q["ler"].mean(), f"{lr['pi'][0]:.2f} … {lr['pi'][1]:.2f}", "нм",
               lr["pi"][0] <= q["ler"].mean() <= lr["pi"][1])


def test_new_lines_fall_into_prediction_interval(rec, plain):
    """Новые независимые линии попадают в интервал предсказания с заявленной вероятностью.

    Метод: 48 новых реализаций с генераторами, не использованными в оценке; доля значений CD и LER внутри
    интервала предсказания (P = 0,95).
    Критерий: доля не меньше 0,85 (при точно известных границах и вероятности 0,95 меньшая доля на 48 линиях
    встречается в 0,2 % случаев; запас учитывает то, что границы оценены всего по 16 линиям).
    Зачем: главное свойство прогноза — указанный диапазон действительно содержит будущие результаты.
    """
    r = simulate(replace(P, seed=4242), 48, keep_map=False)
    q = per_line(r, P)
    for key in ("cd", "ler"):
        lo, hi = plain["quantities"][key]["total"]["pi"]
        share = float(np.mean((q[key] >= lo) & (q[key] <= hi)))
        assert rec(f"Доля новых линий внутри интервала: {key.upper()}", share, "≥ 0,85", "", share >= 0.85)


def test_edge_points_distribution(rec, plain):
    """Отклонения точек края близки к нормальным: доля внутри x̄ ± zσ равна доверительной вероятности.

    Метод: все точки краёв 16 реализаций (9600 значений), P = 0,95.
    Критерий: доля внутри полосы 0,95 ± 0,01; выборочные квантили 2,5 % и 97,5 % отличаются от x̄ ∓ 1,96σ не более чем на 0,1σ.
    Зачем: если распределение отклонений нормальное, полоса x̄ ± zσ — корректный прогноз диапазона положения края.
    """
    ps = plain["point_stats"]
    assert rec("Доля точек внутри полосы", ps["inside"], "0,95 ± 0,01", "", abs(ps["inside"] - 0.95) <= 0.01)
    d = max(abs(ps["q"][0] - ps["band"][0]), abs(ps["q"][1] - ps["band"][1])) / ps["sd"]
    assert rec("Расхождение квантилей с полосой, в долях σ", d, "≤ 0,1", "", d <= 0.1)


def test_process_variation_widens_intervals(rec, plain, varied):
    """Нестабильность дозы и фокуса расширяет интервал предсказания CD и почти не влияет на LER.

    Метод: та же оценка с СКО дозы 3 % и фокуса 40 нм; общие случайные числа с расчётом без нестабильности.
    Критерий: СКО CD растёт больше чем вдвое; вклад нестабильности в разброс CD > 50 %; в разброс CD он больше,
    чем в разброс LER; разброс без нестабильности совпадает с отдельной оценкой без неё в пределах 2 раз.
    Зачем: ширина линии определяется дозой и фокусом, а шероховатость — в основном дробовым шумом;
    разложение разброса должно это показывать, иначе технолог будет бороться не с той причиной.
    """
    a, b = plain["quantities"], varied["quantities"]
    r = b["cd"]["total"]["sd"] / a["cd"]["total"]["sd"]
    assert rec("Рост СКО CD", r, "> 2", "раз", r > 2)
    assert rec("Вклад нестабильности в разброс CD", b["cd"]["process_share"], "> 0,5", "", b["cd"]["process_share"] > 0.5)
    assert rec("Вклад нестабильности в разброс LER", b["ler"]["process_share"], f"< {b['cd']['process_share']:.2f}", "",
               b["ler"]["process_share"] < b["cd"]["process_share"])
    k = b["cd"]["stoch"]["sd"] / a["cd"]["total"]["sd"]
    assert rec("СКО CD без нестабильности: отношение двух оценок", k, "0,5 … 2", "", 0.5 <= k <= 2)


def test_reproducible_and_cancellable(rec):
    """Оценка воспроизводима при том же генераторе и останавливается по запросу.

    Метод: два одинаковых запуска с нестабильностью дозы; третий запуск с немедленной остановкой.
    Критерий: результаты совпадают побитово; остановленный запуск возвращает «нет результата».
    Зачем: повторный расчёт для отчёта должен дать те же числа; остановка не должна выдавать неполный результат за полный.
    """
    a = estimate(P, 4, 0.9, dict(dose=2.0))
    b = estimate(P, 4, 0.9, dict(dose=2.0))
    same = all(np.array_equal(a["quantities"][k]["values"], b["quantities"][k]["values"]) for k in a["quantities"])
    assert rec("Совпадение двух запусков", same, "да", "", same)
    c = estimate(P, 4, 0.9, dict(dose=2.0), cancel=lambda: True)
    assert rec("Результат остановленного запуска", "нет" if c is None else "есть", "нет", "", c is None)
