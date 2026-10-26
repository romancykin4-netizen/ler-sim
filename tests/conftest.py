import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from lersim.process import Process
from tests import report

nm = 1e-9
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def p():
    """Рабочая точка по умолчанию на коротком участке 0,6 мкм, чтобы тесты шли быстро."""
    return replace(Process(), L=600.0)


@pytest.fixture
def rec(request):
    """Запись измерения в отчёт; возвращает признак «в норме», чтобы условие писалось один раз:
    assert rec("Отклонение", err, "< 1e-12", "нм", err < 1e-12)."""
    def r(name, value, expect="", unit="", ok=None):
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = f"массив {value.size} значений, от {report.fmt(float(value.min()))} до {report.fmt(float(value.max()))}"
        ok = None if ok is None else bool(ok)
        request.node.user_properties.append(("rec", dict(name=name, value=value, expect=expect, unit=unit, ok=ok)))
        return True if ok is None else ok
    return r


def se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(x.size)


def ler(r):
    return 0.5 * (r["m"]["LER_L"] + r["m"]["LER_R"])


_col = report.Collector()
_t0 = [0.0]


def pytest_sessionstart(session):
    _t0[0] = time.perf_counter()


def pytest_runtest_logreport(report):
    _col.add(report)


def pytest_sessionfinish(session):
    """Полный прогон (оба набора, без отбора -k/-m) — отчёт HTML и CSV; частичный — краткий текстовый."""
    rows = _col.rows(session)
    if not rows:
        return
    from datetime import datetime
    out = ROOT / "logs"
    out.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y-%m-%d_%H%M%S}"
    opt = session.config.option
    kinds = {r["kind"] for r in rows}
    full = kinds >= set(report.KIND.values()) and not getattr(opt, "keyword", "") and not getattr(opt, "markexpr", "")
    if full:
        html_path = out / f"отчёт_тестов_{stamp}.html"
        csv_path = out / f"отчёт_тестов_{stamp}.csv"
        mcsv = report.write_csv(rows, csv_path)
        report.write_html(rows, html_path, time.perf_counter() - _t0[0], csv_path, mcsv)
        session.config._report_paths = [html_path, csv_path, mcsv]
    else:
        txt = out / f"отчёт_тестов_{stamp}.txt"
        report.write_txt(rows, txt)
        session.config._report_paths = [txt]


def pytest_terminal_summary(terminalreporter, config):
    for path in getattr(config, "_report_paths", []):
        terminalreporter.write_line(f"Отчёт: {path}")
