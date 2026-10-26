"""Анализ компьютера и выбор числа потоков (lersim/hw.py); параллельный расчёт точек окна процесса."""
from dataclasses import replace
import numpy as np
import pytest
from lersim import hw
from lersim.process import Process, simulate, _lercore
from lersim.sweep import window


def test_thread_choice_rules(rec):
    """Число потоков: все ядра, кроме одного для интерфейса; ограничение по памяти; минимум один поток.

    Метод: правило выбора на заданных конфигурациях: 2, 4, 12 ядер при 16 ГБ свободной памяти; 12 ядер при 1 ГБ.
    Критерий: 2 → 2; 4 → 3; 12 → 11; при 1 ГБ — не больше, чем помещается в половину памяти при наибольшей сетке,
    но не меньше одного.
    Зачем: на слабом компьютере расчёт не должен вытеснять интерфейс и исчерпывать память.
    """
    for cores, mem, exp in ((2, 16, 2), (4, 16, 3), (12, 16, 11)):
        t = hw.choose_threads(cores, mem)[0]
        assert rec(f"{cores} ядер, {mem} ГБ", t, f"{exp}", "", t == exp)
    t = hw.choose_threads(12, 1.0)[0]
    cap = max(1, int(0.5 * 2**30 // (hw.BYTES_PER_CELL * hw.MAX_CELLS)))
    assert rec("12 ядер, 1 ГБ", t, f"{cap}", "", t == cap)
    assert rec("12 ядер, 0,01 ГБ", hw.choose_threads(12, 0.01)[0], "1", "", hw.choose_threads(12, 0.01)[0] == 1)


def test_analysis_reports_machine(rec):
    """Анализ возвращает конфигурацию и замер скорости, пригодные для оценки времени.

    Метод: автоматический анализ и анализ с ручным числом потоков 1.
    Критерий: ядер ≥ 1, физических ≤ логических, память > 0, времена > 0; ручное значение применяется
    к вычислительному ядру; оценка времени растёт с числом реализаций.
    Зачем: по этим данным программа выбирает потоки и сообщает ожидаемое время долгих расчётов.
    """
    m = hw.analyze()
    rec("Логических ядер; физических", f"{m.logical}; {m.physical}")
    assert rec("Физических не больше логических", m.physical <= m.logical, "да", "", 1 <= m.physical <= m.logical)
    assert rec("Свободная память", m.mem_free, "> 0", "ГБ", m.mem_free > 0)
    assert rec("Время одной реализации", m.t_single, "> 0", "с", m.t_single > 0 and m.t_task > 0)
    assert rec("Оценка времени растёт с числом реализаций", m.estimate(100) > m.estimate(10), "да", "", m.estimate(100) > m.estimate(10))
    hw.analyze(1)
    if _lercore is not None:
        assert rec("Ручное значение 1 применено к ядру C++", _lercore.threads(), "1", "", _lercore.threads() == 1)
    hw.analyze()


@pytest.mark.skipif(_lercore is None, reason="модуль C++ не собран")
def test_result_independent_of_thread_setting(rec):
    """Результат побитово одинаков при 1 потоке и при автоматическом выборе.

    Метод: один и тот же расчёт (3 реализации, участок 0,6 мкм) после set_threads(1) и после автоматического анализа.
    Критерий: суммы координат краёв совпадают побитово.
    Зачем: анализатор меняет только скорость; научный результат не должен зависеть от компьютера.
    """
    p = replace(Process(), L=600.0)
    _lercore.set_threads(1)
    a = simulate(p, 3, keep_map=False)
    hw.analyze()
    b = simulate(p, 3, keep_map=False)
    same = np.array_equal(a["xL"], b["xL"]) and np.array_equal(a["xR"], b["xR"])
    assert rec("Края совпадают побитово", same, "да", "", same)


@pytest.mark.skipif(_lercore is None, reason="модуль C++ не собран")
def test_window_batch_equals_single_runs(rec):
    """Пакетный параллельный расчёт окна процесса даёт те же числа, что отдельные расчёты точек.

    Метод: сетка 3 дозы × 2 фокуса пакетом; каждая точка отдельно функцией simulate с тем же генератором.
    Критерий: CD совпадает побитово во всех точках.
    Зачем: ускорение за счёт параллельного счёта точек не должно менять результат.
    """
    p = replace(Process(), L=600.0)
    doses, focuses = np.array([30.0, 32.5, 35.0]), np.array([0.0, 100.0])
    CD, _ = window(p, doses, focuses)
    ref = np.array([[simulate(replace(p, dose=d, focus=f), 1, keep_map=False)["m"]["CD"] for f in focuses] for d in doses])
    same = np.array_equal(CD, ref)
    assert rec("CD пакетом и по точкам совпадают", same, "да", "", same)
