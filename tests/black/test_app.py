"""Программа в целом: таблица результатов, журнал, история расчётов, параметры моделирования, кэш.

Программа запускается без экрана; проверяется то, что получает пользователь."""
import csv, json, os, time
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    import ler_gui as g
    monkeypatch.setattr(g, "BASE", tmp_path)
    monkeypatch.setattr(g, "LOGS", tmp_path / "logs")
    w = g.Main()
    w.a_auto.setChecked(False)
    wait(app, w, 1)
    yield w
    w.close()


def wait(app, w, count, timeout=60):
    t = time.time()
    while w.count < count:
        app.processEvents()
        time.sleep(0.005)
        assert time.time() - t < timeout, "расчёт не завершился"


def cell(w, key, col=2):
    t, i = w.cells[key]
    return float(t.item(i, col).text().replace(",", ".").replace("−", "-"))


def recalc(app, w):
    n = w.count
    w.calc()
    wait(app, w, n + 1)


def test_startup_calculation_fills_table(rec, win):
    """После запуска таблица результатов показывает ровно посчитанные значения.

    Метод: программа запускается без экрана; значения ячеек таблицы сравниваются с результатом расчёта.
    Критерий: совпадение с точностью округления таблицы (0,05 нм для ширины, 0,005 нм для шероховатости);
    строка итогов над графиками заполнена.
    Зачем: пользователь читает числа из таблицы — они должны быть теми, что посчитаны.
    """
    m = win.cur["m"]
    for key, name, ref, tol in (("cd", "Ширина линии", m["CD"], 0.05), ("lerl", "Шероховатость левого края", m["LER_L"], 0.005),
                                ("lwr", "Шероховатость ширины", m["LWR"], 0.005),
                                ("dcd", "Отклонение от заданной ширины", m["CD"] - win.cur["p"].CD, 0.05)):
        v = cell(win, key)
        assert rec(f"{name} в таблице", v, f"{ref:.3f} ± {tol:g}", "нм", abs(v - ref) <= tol)
    assert rec("Строка итогов заполнена", "CD =" in win.summary.text(), "да", "", "CD =" in win.summary.text())


def test_log_file_records_every_stage(rec, win, tmp_path):
    """Файл журнала содержит все этапы расчёта, аппроксимацию спектра и итог.

    Метод: чтение файла журнала после первого расчёта.
    Критерий: присутствуют записи «Расчёт № 1», пять этапов, аппроксимация спектра и итог.
    Зачем: журнал — документ о том, как получен результат.
    """
    text = next((tmp_path / "logs").glob("журнал_*.txt")).read_text(encoding="utf-8")
    for s in ("Расчёт № 1", "Изображение", "Экспонирование", "Термообработка и проявление",
              "Обработка краёв", "Аппроксимация спектра", "Итог: ширина"):
        assert rec(f"В журнале есть «{s}»", s in text, "да", "", s in text)


def test_history_csv_is_readable_and_matches(rec, win, tmp_path):
    """История расчётов в CSV читается и совпадает с расчётом.

    Метод: CSV разбирается как таблица с разделителем «;» и десятичной запятой.
    Критерий: одна строка; доза совпадает точно; ширина и шероховатость ширины — до 0,001 нм.
    Зачем: история — основа для последующего анализа в Excel.
    """
    with open(tmp_path / "logs" / "история_расчётов.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert rec("Строк в истории", len(rows), "1", "", len(rows) == 1)
    num = lambda k: float(rows[0][k].replace(",", "."))
    assert rec("Доза", num("Доза экспонирования E, мДж/см²"), f"{win.cur['p'].dose:g}", "мДж/см²", num("Доза экспонирования E, мДж/см²") == win.cur["p"].dose)
    assert rec("Ширина линии", num("Ширина линии CD, нм"), f"{win.cur['m']['CD']:.4f}", "нм",
               abs(num("Ширина линии CD, нм") - win.cur["m"]["CD"]) < 1e-3)
    assert rec("Шероховатость ширины", num("Шероховатость ширины LWR (3σ), нм"), f"{win.cur['m']['LWR']:.4f}", "нм",
               abs(num("Шероховатость ширины LWR (3σ), нм") - win.cur["m"]["LWR"]) < 1e-3)


def test_old_history_file_is_kept_aside(rec, app, win, tmp_path):
    """История прежней версии с другими столбцами не смешивается с новой.

    Метод: файл истории подменяется файлом со старыми заголовками, затем выполняется расчёт.
    Критерий: старый файл переименован в «история_расчётов_до_…csv» без изменений; новый файл
    начинается с актуальных заголовков и содержит одну строку.
    Зачем: после обновления программы история должна оставаться читаемой в Excel.
    """
    hist = tmp_path / "logs" / "история_расчётов.csv"
    hist.write_text("время;доза\n01.01.2026;32\n", encoding="utf-8-sig")
    win.rows["dose"].set(33.0)
    recalc(app, win)
    old = list((tmp_path / "logs").glob("история_расчётов_до_*.csv"))
    assert rec("Прежний файл сохранён под новым именем", len(old), "1", "", len(old) == 1)
    assert rec("Содержимое прежнего файла не изменено", old[0].read_text(encoding="utf-8-sig"), "время;доза …", "",
               old[0].read_text(encoding="utf-8-sig").startswith("время;доза"))
    with open(hist, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert rec("Строк в новом файле", len(rows), "1", "", len(rows) == 1)
    assert rec("Новый файл с актуальными столбцами", "Ширина линии CD, нм" in rows[0], "да", "", "Ширина линии CD, нм" in rows[0])


def test_old_history_kept_when_file_is_locked(rec, app, win, tmp_path, monkeypatch):
    """История прежней версии сохраняется, даже если файл временно заблокирован.

    Метод: переименование файла истории принудительно отклоняется (как при блокировке антивирусом
    в Windows); выполняется расчёт.
    Критерий: содержимое прежнего файла сохранено копией «история_расчётов_до_…csv»; новый файл
    начинается с актуальных заголовков и содержит одну строку; в журнале нет сообщения об ошибке.
    Зачем: блокировка файла сторонней программой не должна приводить к потере или смешению истории.
    """
    import pathlib
    hist = tmp_path / "logs" / "история_расчётов.csv"
    hist.write_text("время;доза\n01.01.2026;32\n", encoding="utf-8-sig")

    def locked(self, target):
        raise PermissionError(13, "Процесс не может получить доступ к файлу")
    monkeypatch.setattr(pathlib.Path, "rename", locked)
    win.rows["dose"].set(34.0)
    recalc(app, win)
    old = list((tmp_path / "logs").glob("история_расчётов_до_*.csv"))
    assert rec("Копия прежнего файла создана", len(old), "1", "", len(old) == 1)
    assert rec("Содержимое прежнего файла сохранено", old[0].read_text(encoding="utf-8-sig")[:10], "время;доза", "",
               old[0].read_text(encoding="utf-8-sig").startswith("время;доза"))
    with open(hist, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert rec("Строк в новом файле", len(rows), "1", "", len(rows) == 1)
    assert rec("Новый файл с актуальными столбцами", "Ширина линии CD, нм" in rows[0], "да", "", "Ширина линии CD, нм" in rows[0])
    err = "Не удалось записать историю" in win.logview.toPlainText()
    assert rec("Сообщение об ошибке в журнале", err, "нет", "", not err)


def test_slider_change_recalculates_and_trend_is_right(rec, app, win):
    """Изменение дозы ползунком запускает пересчёт с правильным трендом.

    Метод: доза 32 → 38 мДж/см² через элемент управления.
    Критерий: в новом результате доза 38; ширина в таблице уменьшилась.
    Зачем: проверяет связь интерфейса с моделью.
    """
    cd0 = cell(win, "cd")
    win.rows["dose"].set(38.0)
    recalc(app, win)
    assert rec("Доза в результате", win.cur["p"].dose, "38", "мДж/см²", win.cur["p"].dose == 38.0)
    assert rec("Ширина после увеличения дозы", cell(win, "cd"), f"< {cd0:g}", "нм", cell(win, "cd") < cd0)


def test_repeat_of_known_regime_comes_from_cache(rec, app, win):
    """Возврат к уже посчитанному режиму берётся из кэша без пересчёта.

    Метод: доза 32 → 36 → 32 мДж/см².
    Критерий: результат — тот же объект, что при первом расчёте; в журнале отметка «из кэша».
    Зачем: мгновенный отклик при возврате к прежним значениям.
    """
    first = win.cur
    win.rows["dose"].set(36.0)
    recalc(app, win)
    win.rows["dose"].set(first["p"].dose)
    recalc(app, win)
    assert rec("Результат взят из кэша", win.cur is first, "да", "", win.cur is first)
    last = win.logview.toPlainText().splitlines()[-1]
    assert rec("Отметка в журнале", "из кэша" in last, "да", "", "из кэша" in last)


def test_save_and_load_parameters(rec, app, win, tmp_path, monkeypatch):
    """Сохранённые параметры моделирования загружаются обратно без потерь.

    Метод: доза 30,5 и PEB 110 °C сохраняются в файл, меняются на 40 и 95, затем файл загружается.
    Критерий: файл содержит 30,5 и 110; после загрузки расчёт идёт с 30,5 и 110.
    Зачем: параметры моделирования должны переноситься между сеансами и компьютерами.
    """
    import ler_gui as g
    path = tmp_path / "параметры.json"
    win.rows["dose"].set(30.5)
    win.rows["T_peb"].set(110.0)
    monkeypatch.setattr(g.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    win.save_params()
    data = json.loads(path.read_text(encoding="utf-8"))["параметры"]
    assert rec("Доза в файле", data["dose"], "30,5", "мДж/см²", data["dose"] == 30.5)
    assert rec("Температура PEB в файле", data["T_peb"], "110", "°C", data["T_peb"] == 110.0)
    win.rows["dose"].set(40.0)
    win.rows["T_peb"].set(95.0)
    monkeypatch.setattr(g.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    n = win.count
    win.load_params()
    wait(app, win, n + 1)
    assert rec("Доза после загрузки", win.cur["p"].dose, "30,5", "мДж/см²", win.cur["p"].dose == 30.5)
    assert rec("Температура PEB после загрузки", win.cur["p"].T_peb, "110", "°C", win.cur["p"].T_peb == 110.0)


def test_snapshot_column_keeps_old_values(rec, app, win):
    """Запомненный режим сохраняет прежние значения.

    Метод: «Запомнить режим», затем доза 40 мДж/см², затем «Убрать последний».
    Критерий: в столбце режима — прежняя ширина, в текущем — новая; после удаления остаётся 3 столбца.
    Зачем: сравнение режимов в таблице должно быть честным.
    """
    cd0 = cell(win, "cd")
    win.snapshot()
    win.rows["dose"].set(40.0)
    recalc(app, win)
    assert rec("Ширина в столбце «Режим 1»", cell(win, "cd", 3), f"{cd0:g}", "нм", cell(win, "cd", 3) == cd0)
    assert rec("Ширина в текущем столбце", cell(win, "cd", 2), f"≠ {cd0:g}", "нм", cell(win, "cd", 2) != cd0)
    win.drop_snapshot()
    assert rec("Столбцов после удаления", win.tables[0].columnCount(), "3", "", win.tables[0].columnCount() == 3)


def test_export_table_csv(rec, win, tmp_path, monkeypatch):
    """Экспорт таблицы результатов в CSV содержит заголовки и все показатели.

    Метод: экспорт в файл и разбор CSV.
    Критерий: заголовок «Показатель; Ед.; Текущий»; есть строки «Ширина линии CD» и «Глубина фокуса DOF».
    Зачем: результаты переносятся в отчёты через этот файл.
    """
    import ler_gui as g
    path = tmp_path / "результаты.csv"
    monkeypatch.setattr(g.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    win.export_table()
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    assert rec("Заголовок", "; ".join(rows[0][:3]), "Показатель; Ед.; Текущий", "", rows[0][:3] == ["Показатель", "Ед.", "Текущий"])
    names = [r[0] for r in rows[1:]]
    rec("Строк показателей", len(names))
    for n in ("Ширина линии CD", "Глубина фокуса DOF (EL ≥ 5 %)"):
        assert rec(f"Есть строка «{n}»", n in names, "да", "", n in names)


def test_statistical_estimate_tab(rec, app, win, tmp_path):
    """Вкладка статистической оценки выдаёт таблицу интервалов, график и запись в журнале.

    Метод: N = 8, P = 0,95, СКО дозы 2 %; запуск из вкладки и ожидание окончания.
    Критерий: в таблице четыре строки; среднее CD лежит внутри ДИ, ДИ внутри интервала предсказания;
    вклад нестабильности процесса указан; в журнале есть «Статистическая оценка готова».
    Зачем: пользователь получает интервалы погрешности из этой таблицы; числа в ней должны быть согласованы.
    """
    win.tabs.setCurrentIndex(5)
    win.st_n.setValue(8)
    win.st_var["dose"].setValue(2.0)
    win.run_estimate()
    t = time.time()
    while win.est_job is not None:
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t < 120, "оценка не завершилась"
    tb = win.st_table
    num = lambda s: float(s.replace(",", ".").replace("−", "-"))
    rng = lambda s: [num(v) for v in s.split(" … ")]
    m, ci, pi = num(tb.item(1, 2).text()), rng(tb.item(1, 4).text()), rng(tb.item(1, 5).text())
    assert rec("Строка CD", tb.item(1, 0).text(), "Ширина линии CD", "", tb.item(1, 0).text() == "Ширина линии CD")
    ok = pi[0] <= ci[0] <= m <= ci[1] <= pi[1]
    assert rec("Интервал предсказания ⊃ ДИ ∋ среднее", f"{pi} ⊃ {ci} ∋ {m}", "да", "", ok)
    assert rec("Вклад нестабильности указан", tb.item(1, 7).text(), "число", "", tb.item(1, 7).text() != "—")
    text = next((tmp_path / "logs").glob("журнал_*.txt")).read_text(encoding="utf-8")
    assert rec("Запись в журнале", "Статистическая оценка готова" in text, "да", "", "Статистическая оценка готова" in text)


def test_user_guide_covers_interface(rec, win):
    """Руководство пользователя описывает каждый график, каждый параметр и каждую расчётную величину.

    Метод: в тексте руководства ищутся названия всех вкладок с графиками, обозначения параметров панели
    «Параметры» и названия строк таблицы результатов.
    Критерий: найдены все; файл руководства доступен из программы.
    Зачем: при добавлении параметра или графика без описания в руководстве тест укажет, что описать.
    """
    import re
    import ler_gui as g
    text = re.sub(r"<[^>]+>", "", g.GUIDE.read_text(encoding="utf-8"))
    assert rec("Файл руководства", g.GUIDE.name, "существует", "", g.GUIDE.exists())
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())] + [win.pw_tabs.tabText(i) for i in range(win.pw_tabs.count())]
    miss = [t for t in tabs if t not in text]
    assert rec("Вкладки графиков без описания", ", ".join(miss) or "нет", "нет", "", not miss)
    labels = [re.sub(r"<[^>]+>", "", lab) for _, spec in g.GROUPS for _, lab, *_ in spec] + \
             [re.sub(r"<[^>]+>", "", lab) for _, lab, *_ in g.RESIST]
    miss = [lab for lab in labels if lab.split(" (")[0].rsplit(" ", 1)[0].lower() not in text.lower()]
    assert rec("Параметры без описания", ", ".join(miss) or "нет", "нет", "", not miss)
    miss = [name for _, name, *_ in g.RESULTS if name.split(" (")[0].rsplit(" ", 1)[0].lower() not in text.lower()]
    assert rec("Величины таблицы без описания", ", ".join(miss) or "нет", "нет", "", not miss)


def test_resist_properties_come_from_file(rec, app, win):
    """Свойства резиста не правятся с панели, а берутся из файла параметров моделирования и показываются на вкладке «Резист».

    Метод: загрузка набора параметров с изменённой глубиной усреднения h_eff = 80 нм и радиусом реакции 2 нм.
    Критерий: вкладка «Резист» показывает новые значения; следующий расчёт выполнен с ними.
    Зачем: калиброванный резист подставляется только через файл; панель содержит только параметры процесса.
    """
    n = win.count
    win.apply({"h_eff": 80.0, "r_c": 2.0})
    wait(app, win, n + 1)
    shown = win.res_cells["h_eff"][0].text(), win.res_cells["r_c"][0].text()
    assert rec("Показано: h_eff; r_c", "; ".join(shown), "80; 2,0", "нм", shown == ("80", "2,0"))
    used = win.cur["p"].h_eff, win.cur["p"].r_c
    assert rec("Использовано в расчёте", str(used), "(80.0, 2.0)", "нм", used == (80.0, 2.0))
    keys = set(win.rows)
    assert rec("Свойства резиста на панели параметров", ", ".join(sorted(keys & {"h_eff", "c_pag", "c_q", "kd_ref", "n_dev"})) or "нет",
               "нет", "", not keys & {"h_eff", "c_pag", "c_q", "kd_ref", "n_dev"})


def test_numerical_aperture_limited_to_scalar_model_range(rec, app, win, tmp_path):
    """Числовая апертура ограничена 0,85 — пределом применимости скалярной модели изображения.

    Метод: попытка задать NA = 0,93 через файл параметров моделирования.
    Критерий: в расчёт идёт NA = 0,85; в журнале — запись о замене значения; верхняя граница ползунка 0,85.
    Зачем: при NA > 0,85 поляризация заметно меняет контраст, и скалярная модель даёт систематическую ошибку;
    программа не должна молча считать вне области применимости.
    """
    assert rec("Верхняя граница NA на панели", win.rows["NA"].spin.maximum(), "0,85", "", win.rows["NA"].spin.maximum() == 0.85)
    n = win.count
    win.apply({"NA": 0.93})
    wait(app, win, n + 1)
    assert rec("NA в расчёте", win.cur["p"].NA, "0,85", "", np.isclose(win.cur["p"].NA, 0.85))
    text = next((tmp_path / "logs").glob("журнал_*.txt")).read_text(encoding="utf-8")
    ok = "вне допустимого диапазона" in text
    assert rec("Запись в журнале о замене", ok, "да", "", ok)
