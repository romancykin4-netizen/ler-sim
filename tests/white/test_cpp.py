"""Вычислительное ядро C++ (lercore/lercore.cpp): воспроизводимость и согласие с эталонной реализацией."""
import os, subprocess, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from lersim.process import simulate, mean_field, _lercore
from tests.conftest import se

pytestmark = pytest.mark.skipif(_lercore is None, reason="модуль C++ не собран")
ROOT = Path(__file__).resolve().parents[2]


def test_reproducible(rec, p):
    """Ядро C++ при том же генераторе даёт побитово тот же результат.

    Метод: два расчёта по 3 повторения с одинаковыми параметрами.
    Критерий: все контуры краёв совпадают побитово.
    Зачем: параллельный расчёт не должен вносить недетерминированность.
    """
    a, b = simulate(p, 3, "cpp"), simulate(p, 3, "cpp")
    same = np.array_equal(a["xL"], b["xL"]) and np.array_equal(a["xR"], b["xR"])
    assert rec("Контуры совпадают побитово", same, "да", "", same)


def test_result_does_not_depend_on_thread_count(rec, p):
    """Результат не зависит от числа потоков.

    Метод: один и тот же расчёт в отдельных процессах с OMP_NUM_THREADS = 1 и 3; сравнивается сумма всех координат краёв.
    Критерий: суммы совпадают побитово.
    Зачем: у каждой строки свой поток случайных чисел; на машине с другим числом ядер результат обязан быть тем же.
    """
    code = ("from dataclasses import replace; from lersim.process import Process, simulate;"
            "r = simulate(replace(Process(), L=600.0), 3, 'cpp'); print(repr(float(r['xL'].sum() + r['xR'].sum())))")
    out = []
    for n in ("1", "3"):
        env = dict(os.environ, OMP_NUM_THREADS=n, PYTHONPATH=str(ROOT))
        out.append(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                  text=True, check=True).stdout.strip())
        rec(f"Контрольная сумма при {n} потоках", out[-1])
    assert rec("Суммы совпадают", out[0] == out[1], "да", "", out[0] == out[1])


def test_noise_free_limit_matches_mean_field(rec, p):
    """В пределе без дробового шума ядро C++ даёт ширину средне-полевой модели.

    Метод: эффективная толщина 10⁵ нм — шум в 45 раз меньше обычного.
    Критерий: ширина отличается от средне-полевой меньше чем на 0,1 нм; LER < 0,3 нм.
    Зачем: проверяет детерминированную часть ядра C++ (реакция, диффузия, проявление) без влияния шума.
    """
    r = simulate(replace(p, h_eff=1e5), 1, "cpp")
    d = r["m"]["CD"] - mean_field(p)[0]
    assert rec("Ширина: C++ − средне-полевая", d, "|…| < 0,1", "нм", abs(d) < 0.1)
    assert rec("Шероховатость левого края", r["m"]["LER_L"], "< 0,3", "нм", r["m"]["LER_L"] < 0.3)


def test_statistics_match_numpy(rec, p):
    """Статистика ядра C++ совпадает с эталоном NumPy.

    Метод: по 24 повторения каждым ядром; сравниваются средняя ширина и средняя σ края.
    Критерий: разность ширин < 4 стандартных ошибок + 0,3 нм; разность σ < 4 стандартных ошибок.
    Зачем: генераторы случайных чисел разные, поэтому сравнивается статистика, а не контуры.
    """
    out = {}
    for eng in ("cpp", "numpy"):
        r = simulate(p, 24, eng)
        out[eng] = ((r["xR"] - r["xL"]).mean(1), r["sigma_single"])
    (wc, sc), (wn, sn) = out["cpp"], out["numpy"]
    tw, ts = 4 * np.hypot(se(wc), se(wn)) + 0.3, 4 * np.hypot(se(sc), se(sn))
    rec("Средняя ширина, C++", wc.mean(), "", "нм")
    rec("Средняя ширина, NumPy", wn.mean(), "", "нм")
    assert rec("Разность средних ширин", wc.mean() - wn.mean(), f"|…| < {tw:.2f}", "нм", abs(wc.mean() - wn.mean()) < tw)
    rec("Средняя σ края, C++", sc.mean(), "", "нм")
    rec("Средняя σ края, NumPy", sn.mean(), "", "нм")
    assert rec("Разность средних σ", sc.mean() - sn.mean(), f"|…| < {ts:.3f}", "нм", abs(sc.mean() - sn.mean()) < ts)
