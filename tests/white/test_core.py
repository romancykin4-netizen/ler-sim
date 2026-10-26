"""Метрики краёв, спектральный анализ, синтез шероховатости и аппроксимация спектра (lersim/core.py)."""
import numpy as np
import pytest
from lersim.core import (freqs, detrend, edge_metrics, pair_spectra, band_rms, psd_shape, synthesize_edges,
                         make_line, least_squares, fit_psd, noise_correct_psd, diffuse_periodic,
                         integrated_diffusion_transfer, smooth_periodic, transfer_spectrum,
                         arrhenius, development_rate)

nm = 1e-9
dy = 1 * nm


def test_detrend_removes_polynomial_exactly(rec):
    """Удаление среднего и тренда точно убирает полином заданной степени и ничего больше.

    Метод: к синусоиде добавлен полином 3 + 2t − 5t²; после удаления тренда степени 0, 1, 2 остаток
    раскладывается по тем же полиномам, а сумма «остаток + тренд» сравнивается с исходным сигналом.
    Критерий: коэффициенты полинома в остатке не больше 1e-12; сумма восстанавливает сигнал.
    Зачем: LER считается по отклонению края от средней линии; ошибка здесь сдвигает все метрики.
    """
    t = np.linspace(-1, 1, 257)
    base = 3 + 2 * t - 5 * t**2
    noise = np.sin(40 * np.pi * t)
    for order in (0, 1, 2):
        u, b = detrend(base + noise, order)
        c = np.abs(np.polyfit(t, u, order)).max()
        assert rec(f"Остаточный полином степени {order}", c, "≤ 1e-12", "", c <= 1e-12)
        assert rec(f"Сумма остатка и тренда (степень {order}) равна сигналу", bool(np.allclose(u + b, base + noise)), "да", "",
                   np.allclose(u + b, base + noise))


def test_edge_metrics_on_sinusoids(rec):
    """Метрики краёв на синусоидах с известным ответом.

    Метод: левый край — синусоида амплитуды 2, правый — амплитуды 3 со сдвигом фазы φ; целое число периодов.
    Точные значения: σ = a/√2, ρ = cos φ, σ_w² = σ_L² + σ_R² − 2ρσ_Lσ_R, LER = 3σ.
    Критерий: совпадение с точными значениями до 1e-12 при φ = 0, π/3, π/2, π.
    Зачем: проверяет определения LER, LWR и корреляции краёв, на которых построены все результаты.
    """
    y = np.arange(1000)
    k = 2 * np.pi * 5 / 1000
    for phi, name in ((0.0, "0"), (np.pi / 3, "π/3"), (np.pi / 2, "π/2"), (np.pi, "π")):
        m = edge_metrics(-50 + 2.0 * np.sin(k * y), 50 + 3.0 * np.sin(k * y + phi))
        assert rec(f"φ = {name}: ширина", m["CD"], "100", "нм", np.isclose(m["CD"], 100))
        assert rec(f"φ = {name}: σ левого края", m["sigma_L"], f"2/√2 = {2 / np.sqrt(2):.4f}", "нм", np.isclose(m["sigma_L"], 2 / np.sqrt(2)))
        assert rec(f"φ = {name}: σ правого края", m["sigma_R"], f"3/√2 = {3 / np.sqrt(2):.4f}", "нм", np.isclose(m["sigma_R"], 3 / np.sqrt(2)))
        assert rec(f"φ = {name}: корреляция краёв", m["rho"], f"cos φ = {np.cos(phi):.4f}", "", abs(m["rho"] - np.cos(phi)) < 1e-12)
        w2 = m["sigma_L"] ** 2 + m["sigma_R"] ** 2 - 2 * m["rho"] * m["sigma_L"] * m["sigma_R"]
        assert rec(f"φ = {name}: σ ширины", m["sigma_w"], f"√(σL²+σR²−2ρσLσR) = {np.sqrt(w2):.4f}", "нм", np.isclose(m["sigma_w"] ** 2, w2))
        assert np.isclose(m["LER_L"], 3 * m["sigma_L"]) and np.isclose(m["LWR"], 3 * m["sigma_w"])


@pytest.mark.parametrize("N", [1024, 1023])
def test_parseval_even_and_odd(rec, N):
    """Площадь под спектром равна дисперсии края (равенство Парсеваля).

    Метод: случайный край из N точек, односторонний спектр, сумма спектра × шаг частоты против дисперсии.
    Критерий: относительное расхождение не больше 1e-12 для чётного и нечётного N.
    Зачем: нормировка спектра — частая ошибка; без неё σ из спектра и σ по краю не совпадут.
    """
    u = detrend(np.random.default_rng(0).standard_normal(N) * nm)[0]
    S = pair_spectra(u, u, dy)["LL"]
    a, b = band_rms(S, N, dy) ** 2, (u**2).mean()
    assert rec("Относительное расхождение площади спектра и дисперсии", abs(a / b - 1), "≤ 1e-12", "", abs(a / b - 1) <= 1e-12)


def test_width_and_center_spectra_identities(rec):
    """Спектры ширины и центра, собранные из спектров краёв, совпадают с прямым расчётом.

    Метод: S_w = S_LL + S_RR − 2·Re S_LR и S_c = ¼(S_LL + S_RR + 2·Re S_LR) против спектров (uR − uL) и (uL + uR)/2.
    Критерий: относительное совпадение до 1e-10.
    Зачем: спектр ширины в программе строится именно этой формулой.
    """
    uL, uR = np.random.default_rng(1).standard_normal((2, 3, 512)) * nm
    sp = pair_spectra(uL, uR, dy)
    ew = np.abs(sp["w"] / pair_spectra(uR - uL, uR - uL, dy)["LL"] - 1).max()
    c = 0.5 * (uL + uR)
    ec = np.abs(sp["c"][..., 1:] / pair_spectra(c, c, dy)["LL"][..., 1:] - 1).max()
    assert rec("Спектр ширины: максимальное относительное расхождение", ew, "≤ 1e-10", "", ew <= 1e-10)
    assert rec("Спектр центра: максимальное относительное расхождение", ec, "≤ 1e-10", "", ec <= 1e-10)


def test_psd_shape_normalisation(rec):
    """Модельный спектр нормирован: заданная σ в полосе частот получается точно.

    Метод: спектр формы [1 + (2πfξ)²]^−(α+½) с σ = 2 нм, ξ = 20 нм, α = 0,7; σ восстанавливается из площади.
    Критерий: σ = 2 нм с точностью 1e-12, нулевая частота исключена.
    Зачем: генератор синтетических краёв и подгонка спектра опираются на эту нормировку.
    """
    S = psd_shape(freqs(2048, dy), 2 * nm, 20 * nm, 0.7)
    s = band_rms(S, 2048, dy) / nm
    assert rec("Нулевая частота", float(S[0]), "0", "", S[0] == 0)
    assert rec("σ из площади спектра", s, "2", "нм", abs(s - 2) < 1e-12)


@pytest.mark.parametrize("rho,ratio", [(1.0, 0.0), (-1.0, 2.0)])
def test_synthesis_limits_rho_plus_minus_one(rec, rho, ratio):
    """Синфазные края дают нулевую шероховатость ширины, противофазные — удвоенную.

    Метод: синтез пары краёв с корреляцией ρ = ±1, сравнение σ ширины с σ края.
    Критерий: σ_w = 0 при ρ = 1 и σ_w = 2σ при ρ = −1 (точность 1e-15 нм).
    Зачем: предельные случаи формулы LWR = √(2(1−ρ))·LER.
    """
    S = psd_shape(freqs(1024, dy), 2 * nm, 15 * nm, 0.5)
    m = edge_metrics(*synthesize_edges(1024, dy, S, rho, 4, 3))
    assert rec("σ ширины / σ края", m["sigma_w"] / m["sigma_L"], f"{ratio:g}", "", abs(m["sigma_w"] - ratio * m["sigma_L"]) < 1e-15)


def test_synthesis_ensemble_statistics(rec):
    """Генератор шероховатости по ансамблю воспроизводит заданные σ, ρ и спектр.

    Метод: 2000 линий длиной 2 мкм с σ = 2 нм, ξ = 25 нм, α = 0,5, ρ = 0,35; средние по ансамблю.
    Критерий: σ² в пределах 4 стандартных ошибок; ρ в пределах ±0,02; медианное отклонение спектра < 5 %.
    Зачем: подтверждает статистическую корректность синтеза, которым проверяется обработка спектров.
    """
    y, xL, xR, f, S = make_line(100 * nm, 2048 * nm, dy, 2 * nm, 25 * nm, 0.5, 0.35, 2000, 7)
    uL, uR = detrend(xL)[0], detrend(xR)[0]
    v = (uL**2).mean(-1) / nm**2
    se = v.std(ddof=1) / np.sqrt(v.size)
    assert rec("Средняя σ² края", v.mean(), f"4 ± {4 * se:.3f}", "нм²", abs(v.mean() - 4) < 4 * se)
    rho = edge_metrics(xL, xR)["rho"]
    assert rec("Корреляция краёв", rho, "0,35 ± 0,02", "", abs(rho - 0.35) < 0.02)
    band = (f > 0) & (f < 0.1 / nm)
    dev = np.median(np.abs(pair_spectra(uL, uR, dy)["LL"].mean(0)[band] / S[band] - 1))
    assert rec("Медианное отклонение спектра от заданного", 100 * dev, "< 5", "%", dev < 0.05)


def test_least_squares_matches_linear_regression(rec):
    """Собственный метод Левенберга–Марквардта совпадает с точной линейной регрессией.

    Метод: прямая 1,5 − 0,7x с шумом 1 %; решение итерационным методом против точного (нормальные уравнения).
    Критерий: коэффициенты совпадают до 1e-8.
    Зачем: этот метод подгоняет спектр шероховатости; SciPy в программе не используется.
    """
    x = np.linspace(0, 1, 50)
    y = 1.5 - 0.7 * x + 0.01 * np.random.default_rng(2).standard_normal(50)
    p = least_squares(lambda c: c[0] + c[1] * x - y, [0.0, 0.0])
    e = np.abs(p - np.polyfit(x, y, 1)[::-1]).max()
    assert rec("Расхождение коэффициентов с точным решением", e, "≤ 1e-8", "", e <= 1e-8)


def test_fit_recovers_parameters(rec):
    """Подгонка спектра восстанавливает заданные σ, длину корреляции и показатель.

    Метод: 1000 синтетических линий 4 мкм с σ = 2 нм, ξ = 20 нм, α = 0,7; подгонка усреднённого спектра.
    Критерий: σ ±3 %, ξ ±8 %, α ±0,05.
    Зачем: параметры спектра выводятся пользователю; подгонка должна возвращать то, что заложено.
    """
    y, xL, xR, f, S = make_line(100 * nm, 4096 * nm, dy, 2 * nm, 20 * nm, 0.7, 0.0, 1000, 11)
    s, xi, a, _ = fit_psd(f, pair_spectra(detrend(xL)[0], detrend(xR)[0], dy)["LL"].mean(0))
    assert rec("СКО", s / nm, "2 ± 3 %", "нм", abs(s / (2 * nm) - 1) < 0.03)
    assert rec("Длина корреляции", xi / nm, "20 ± 8 %", "нм", abs(xi / (20 * nm) - 1) < 0.08)
    assert rec("Показатель шероховатости", a, "0,7 ± 0,05", "", abs(a - 0.7) < 0.05)


def test_fit_with_floor_separates_white_noise(rec):
    """Подгонка разделяет коррелированную шероховатость и белый шум.

    Метод: край = коррелированная часть (σ = 2, ξ = 15, α = 1) + белый шум σ = 0,7 на шаге 1 нм; 400 линий.
    Ожидаемый уровень белого шума в одностороннем спектре 2·0,7²·1 = 0,98.
    Критерий: σ ±0,08; ξ ±2; α ±0,1; уровень шума ±5 %.
    Зачем: в программе спектр края содержит белый пол; без его учёта показатель α получался нефизичным.
    """
    rng = np.random.default_rng(5)
    f = freqs(4096, 1.0)
    u = synthesize_edges(4096, 1.0, psd_shape(f, 2.0, 15.0, 1.0), 0.0, 400, rng)[0] + 0.7 * rng.standard_normal((400, 4096))
    u = detrend(u)[0]
    s, xi, a, _, floor = fit_psd(f, pair_spectra(u, u, 1.0)["LL"].mean(0), floor=True)
    assert rec("СКО коррелированной части", s, "2 ± 0,08", "нм", abs(s - 2.0) < 0.08)
    assert rec("Длина корреляции", xi, "15 ± 2", "нм", abs(xi - 15) < 2)
    assert rec("Показатель шероховатости", a, "1 ± 0,1", "", abs(a - 1.0) < 0.1)
    assert rec("Уровень белого шума", floor, "0,98 ± 5 %", "нм³", abs(floor / 0.98 - 1) < 0.05)


def test_noise_correction_keeps_negative_bins(rec):
    """Вычитание шума измерения сохраняет отрицательные отсчёты.

    Метод: из спектра [1; 0,5] вычитается шум [0,8; 0,9].
    Критерий: второй отсчёт остаётся отрицательным (−0,4), а не обнуляется.
    Зачем: обнуление отрицательных отсчётов систематически завышает оценку шероховатости.
    """
    v = noise_correct_psd(np.array([1.0, 0.5]), np.array([0.8, 0.9]))[1]
    assert rec("Второй отсчёт после вычитания", v, "−0,4", "", np.isclose(v, -0.4))


def test_linear_transfer_functions(rec):
    """Диффузия, интегральная передача и сглаживание совпадают с аналитическими формулами.

    Метод: косинус с 8 периодами на 256 точках; затухание амплитуды против exp(−D(2πf)²t).
    Критерий: относительная погрешность 1e-10; среднее сохраняется; передача на нулевой частоте = 1;
    матричная передача спектров даёт 0,25 + 0,1 = 0,35.
    Зачем: эти операторы используются при оценке сглаживания шероховатости диффузией.
    """
    N = 256
    y = np.arange(N) * dy
    c = 1 + np.cos(2 * np.pi * 8 * y / (N * dy))
    D, t = 1e-18, 30.0
    g = np.exp(-D * (2 * np.pi * 8 / (N * dy)) ** 2 * t)
    cd = diffuse_periodic(c, dy, D, t)
    assert rec("Амплитуда после диффузии", cd.max() - 1, f"{g:.6f}", "", np.isclose(cd.max() - 1, g, rtol=1e-10))
    assert rec("Среднее после диффузии", cd.mean(), "1", "", np.isclose(cd.mean(), 1))
    assert rec("Интегральная передача на f = 0", float(integrated_diffusion_transfer(np.array([0.0]), D, t)[0]), "1", "",
               integrated_diffusion_transfer(np.array([0.0]), D, t)[0] == 1.0)
    sm = smooth_periodic(c - 1, dy, D, 0.0, t).max()
    assert rec("Амплитуда после сглаживания края", sm, f"{g:.6f}", "", np.isclose(sm, g, rtol=1e-10))
    T = np.broadcast_to(np.eye(2) * 0.5, (4, 2, 2))
    Sin = np.broadcast_to(np.eye(2), (4, 2, 2))
    v = transfer_spectrum(T, Sin, Q=0.1 * Sin)[0, 0, 0]
    assert rec("Матричная передача спектров", v, "0,35", "", np.isclose(v, 0.35))


def test_arrhenius(rec):
    """Закон Аррениуса для скоростей и диффузии.

    Метод: k(T) = k_ref·exp(−E/k_B·(1/T − 1/T_ref)) при T = T_ref, E = 0 и T выше/ниже опорной.
    Критерий: k(T_ref) = k_ref; при E = 0 не зависит от T; при E > 0 растёт с температурой.
    Зачем: так температура PEB влияет на длину диффузии и скорость снятия защиты.
    """
    assert rec("k при опорной температуре", arrhenius(2.0, 0.5, 378.15, 378.15), "2", "", arrhenius(2.0, 0.5, 378.15, 378.15) == 2.0)
    assert rec("k при E = 0 и другой температуре", arrhenius(2.0, 0.0, 400.0, 378.15), "2", "", arrhenius(2.0, 0.0, 400.0, 378.15) == 2.0)
    hi, lo = arrhenius(2.0, 0.5, 390.0, 378.15), arrhenius(2.0, 0.5, 370.0, 378.15)
    assert rec("k при 117 °C", hi, "> 2", "", hi > 2.0)
    assert rec("k при 97 °C", lo, "< 2", "", lo < 2.0)


@pytest.mark.parametrize("n", [1.0, 4.0, 8.0])
def test_development_rate_endpoints_and_monotonic(rec, n):
    """Скорость проявления по модели Мака: пределы и монотонность.

    Метод: R(d) на 141 точке d от −0,2 до 1,2 при R_min = 0,02, R_max = 100 нм/с.
    Критерий: R = R_min при d ≤ 0, R = R_max при d ≥ 1, функция не убывает.
    Зачем: от R(d) зависит положение края; немонотонность дала бы несколько краёв.
    """
    d = np.linspace(-0.2, 1.2, 141)
    R = development_rate(d, 0.02, 100.0, 0.01, n)
    assert rec("R при d ≤ 0", float(R[0]), "0,02", "нм/с", np.isclose(R[d <= 0], 0.02).all())
    assert rec("R при d ≥ 1", float(R[-1]), "100", "нм/с", np.isclose(R[d >= 1], 100.0).all())
    assert rec("Наименьшее приращение R", float(np.diff(R).min()), "≥ 0", "нм/с", (np.diff(R) >= 0).all())
