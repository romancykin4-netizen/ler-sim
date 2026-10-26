import sys, os, json, csv, time, re
import multiprocessing
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from dataclasses import asdict, replace, fields
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt, QLocale, QTimer, QObject, QRunnable, QThreadPool, QThread, Signal, QSettings, QUrl
from PySide6.QtGui import QAction, QKeySequence, QFont, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QSlider, QDoubleSpinBox, QSpinBox,
                               QComboBox, QGridLayout, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea,
                               QPushButton, QCheckBox, QFileDialog, QMessageBox, QTabWidget, QDockWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QPlainTextEdit, QProgressBar,
                               QAbstractItemView, QDialog)
from lersim.process import Process, simulate, _lercore
from lersim.core import fit_psd
from lersim.sweep import window, latitude
from lersim import figs, hw
from lersim.stats import estimate

APP = "Предиктивное моделирование LER при фотолитографии"
VERSION = "1.1"
CACHE_MB = 128
BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
LOGS = BASE / "logs"
GUIDE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "docs" / "руководство.html"

GROUPS = [
    ("Маска", [
        ("CD", "Ширина линии на маске CD<sub>0</sub>", "нм", 80, 240, 1, 0, "Заданная (целевая) ширина тёмной линии в масштабе пластины"),
        ("pitch", "Период решётки p", "нм", 200, 600, 10, 0, "Период линий и пробелов; не меньше ширины линии + 40 нм")]),
    ("Экспонирование", [
        ("NA", "Числовая апертура NA", "", 0.50, 0.85, 0.01, 2,
         "Числовая апертура проекционного объектива. Не более 0,85: модель изображения скалярная, при большей "
         "апертуре поляризационные эффекты существенны"),
        ("sigma_s", "Степень когерентности σ", "", 0.0, 0.95, 0.05, 2, "Радиус круглого источника в долях NA (σ)"),
        ("dose", "Доза экспонирования E", "мДж/см²", 15, 70, 0.5, 1, "Энергия экспонирования на единицу площади"),
        ("focus", "Расфокусировка Δz", "нм", -200, 200, 5, 0, "Расстояние от плоскости наилучшей фокусировки")]),
    ("Постэкспозиционная сушка (PEB)", [
        ("T_peb", "Температура T<sub>PEB</sub>", "°C", 85, 125, 1, 0, "Температура плиты PEB; влияет на диффузию кислоты и скорость реакции"),
        ("t_peb", "Длительность t<sub>PEB</sub>", "с", 30, 120, 1, 0, "Длительность PEB"),
        ("L_D", "Длина диффузии кислоты L<sub>D</sub>", "нм", 5, 60, 0.5, 1,
         "Среднее смещение кислоты √(2·D·t) при 105 °C и 60 с — свойство резиста. При другой температуре и времени "
         "фактическая длина пересчитывается по закону Аррениуса и √t; она показана в таблице результатов")]),
    ("Проявление", [
        ("t_dev", "Длительность t<sub>dev</sub>", "с", 30, 90, 1, 0, "Длительность проявления")]),
]
RESIST = [
    ("c_pag", "Концентрация ФГК [PAG]<sub>0</sub>", "нм⁻³", 3),
    ("c_q", "Концентрация тушителя [Q]<sub>0</sub>", "нм⁻³", 3),
    ("C_g", "Константа Дилла C", "см²/мДж", 3),
    ("h_eff", "Глубина усреднения h<sub>eff</sub>", "нм", 0),
    ("kd_ref", "Константа депротекции k<sub>d</sub> (105 °C)", "нм³/с", 2),
    ("E_D", "Энергия активации диффузии E<sub>D</sub>", "эВ", 2),
    ("E_kd", "Энергия активации депротекции E<sub>kd</sub>", "эВ", 2),
    ("R_max", "Максимальная скорость растворения R<sub>max</sub>", "нм/с", 0),
    ("R_min", "Минимальная скорость растворения R<sub>min</sub>", "нм/с", 2),
    ("n_dev", "Параметр селективности n", "", 1),
    ("d_th", "Пороговая степень депротекции d<sub>th</sub>", "", 3),
    ("r_c", "Радиус реакции r<sub>c</sub>", "нм", 1),
]

PARAM_NAMES = {
    "CD": "Ширина линии на маске CD₀, нм", "pitch": "Период решётки p, нм", "wl": "Длина волны λ, нм",
    "NA": "Числовая апертура NA", "sigma_s": "Степень когерентности σ", "dose": "Доза экспонирования E, мДж/см²",
    "focus": "Расфокусировка Δz, нм", "T_peb": "Температура PEB, °C", "t_peb": "Длительность PEB, с",
    "t_dev": "Длительность проявления, с", "L": "Длина фрагмента линии L, нм", "dx": "Шаг сетки Δx, нм",
    "dy": "Шаг сетки Δy, нм", "seed": "Начальное значение генератора", "c_pag": "Концентрация ФГК [PAG]₀, нм⁻³",
    "c_q": "Концентрация тушителя [Q]₀, нм⁻³", "C_g": "Константа Дилла C, см²/мДж", "h_eff": "Глубина усреднения h_eff, нм",
    "L_D": "Длина диффузии кислоты при опорных условиях, нм", "t_ref": "Опорная длительность PEB, с",
    "kd_ref": "Константа депротекции при опорной температуре, нм³/с", "T_ref": "Опорная температура PEB, °C",
    "E_D": "Энергия активации диффузии, эВ", "E_kd": "Энергия активации депротекции, эВ",
    "R_max": "Максимальная скорость растворения, нм/с", "R_min": "Минимальная скорость растворения, нм/с",
    "n_dev": "Параметр селективности n", "d_th": "Пороговая степень депротекции d_th",
    "r_c": "Радиус реакции r_c, нм",
}

RESULTS = [
    ("cd", "Ширина линии CD", "нм", lambda r: r["m"]["CD"], 1),
    ("dcd", "Отклонение от номинала ΔCD", "нм", lambda r: r["m"]["CD"] - r["p"].CD, 1),
    ("lerl", "Шероховатость левого края LER (3σ)", "нм", lambda r: r["m"]["LER_L"], 2),
    ("lerr", "Шероховатость правого края LER (3σ)", "нм", lambda r: r["m"]["LER_R"], 2),
    ("lwr", "Шероховатость ширины LWR (3σ)", "нм", lambda r: r["m"]["LWR"], 2),
    ("rho", "Коэффициент корреляции краёв ρ", "", lambda r: r["m"]["rho"], 3),
    ("nils", "Нормированный наклон изображения NILS", "", lambda r: r["NILS"], 2),
    ("ldiff", "Длина диффузии кислоты в режиме PEB", "нм", lambda r: r["L_diff"], 1),
    ("fs", "СКО коррелированной составляющей σ", "нм", lambda r: r["fit"][0], 2),
    ("fxi", "Длина корреляции ξ", "нм", lambda r: r["fit"][1], 1),
    ("fa", "Показатель шероховатости α", "", lambda r: r["fit"][2], 2),
    ("fn", "Уровень белого шума S₀", "нм³", lambda r: r["fit"][4], 2),
    ("lost", "Доля разрывов линии", "%", lambda r: 100 * r["lost_frac"], 1),
    ("n", "Число реализаций N", "", lambda r: r["n"], 0),
    ("t", "Время расчёта", "с", lambda r: r["t_total"], 2),
]

QSS = """
QWidget { font-size: 9pt; }
QGroupBox { font-weight: 600; border: 1px solid #c9ced6; border-radius: 3px; margin-top: 9px; padding: 6px 6px 2px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }
QSlider::groove:horizontal { height: 3px; background: #c9ced6; }
QSlider::sub-page:horizontal { background: #1b3a6b; }
QSlider::handle:horizontal { background: #fff; border: 1px solid #1b3a6b; width: 9px; margin: -5px 0; border-radius: 2px; }
QTableWidget { gridline-color: #e1e4e8; }
QHeaderView::section { background: #eef0f3; border: none; border-right: 1px solid #d5d9e0; border-bottom: 1px solid #d5d9e0; padding: 3px 5px; }
QPlainTextEdit { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt; }
QToolBar { spacing: 2px; }
#summary { padding: 6px 8px; font-size: 10pt; }
#warn { background: #fff4ce; border: 1px solid #e0c35a; padding: 4px 8px; }
"""


class Row(QWidget):
    changed = Signal()

    def __init__(self, label, unit, lo, hi, step, dec, value, tip):
        super().__init__()
        self.lo, self.step = lo, step
        self.name = QLabel(label, textFormat=Qt.RichText)
        self.name.setFixedWidth(150)
        self.name.setWordWrap(True)
        self.name.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal, minimum=0, maximum=round((hi - lo) / step))
        self.slider.setValue(round((value - lo) / step))
        self.spin = QDoubleSpinBox(decimals=dec, minimum=lo, maximum=hi, singleStep=step, value=value,
                                   keyboardTracking=False)
        self.spin.setFixedWidth(72)
        self.spin.setAlignment(Qt.AlignRight)
        self.spin.setToolTip(tip)
        u = QLabel(unit)
        u.setFixedWidth(56)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 1, 0, 1)
        for w in (self.name, self.slider, self.spin, u):
            h.addWidget(w)
        self.slider.valueChanged.connect(lambda i: self.spin.setValue(self.lo + i * self.step))
        self.spin.valueChanged.connect(self._sync)

    def _sync(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self.lo) / self.step))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self):
        return self.spin.value()

    def set(self, v):
        self.spin.setValue(v)


def table(rows, heads):
    t = QTableWidget(rows, len(heads))
    t.setHorizontalHeaderLabels(heads)
    t.verticalHeader().hide()
    t.verticalHeader().setDefaultSectionSize(22)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.ContiguousSelection)
    t.setAlternatingRowColors(True)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return t


class Toolbar(NavigationToolbar2QT):
    toolitems = [("Home", "Исходный масштаб", "home", "home"), ("Back", "Предыдущий вид", "back", "back"),
                 ("Forward", "Следующий вид", "forward", "forward"), (None, None, None, None),
                 ("Pan", "Сдвиг: левая кнопка, масштаб: правая", "move", "pan"),
                 ("Zoom", "Увеличить выделенную область", "zoom_to_rect", "zoom"), (None, None, None, None),
                 ("Save", "Сохранить рисунок", "filesave", "save_figure")]


class Plot(QWidget):
    def __init__(self, *extra):
        super().__init__()
        self.canvas = FigureCanvasQTAgg(Figure(layout="constrained"))
        self.fig = self.canvas.figure
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QHBoxLayout()
        bar.addWidget(Toolbar(self.canvas, self))
        bar.addStretch()
        for w in extra:
            bar.addWidget(w)
        bar.addSpacing(8)
        v.addLayout(bar)
        v.addWidget(self.canvas)
        self.canvas.mpl_connect("draw_event", lambda e: setattr(self.fig, "_relayout", False))

    def resizeEvent(self, e):
        self.fig._relayout = True
        self.fig.set_layout_engine("constrained")
        super().resizeEvent(e)

    def draw(self):
        self.canvas.draw_idle()


class Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class Job(QRunnable):
    def __init__(self, rid, p, n, engine, sig):
        super().__init__()
        self.rid, self.p, self.n, self.engine, self.sig = rid, p, n, engine, sig

    def run(self):
        try:
            r = simulate(self.p, self.n, self.engine)
            r["fit"] = fit_psd(r["f"], r["S_edge"], floor=True)
            r["d"] = r["d"].astype(np.float32)
            r["mb"] = sum(v.nbytes for v in r.values() if isinstance(v, np.ndarray)) / 2**20
            r.update(p=self.p, n=self.n)
            self.sig.done.emit(self.rid, r)
        except Exception as e:
            self.sig.failed.emit(self.rid, f"{type(e).__name__}: {e}")


class Sweep(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)

    def __init__(self, p, doses, focuses, engine):
        super().__init__()
        self.p, self.doses, self.focuses, self.engine = p, doses, focuses, engine
        self.stop = False

    def run(self):
        t = time.perf_counter()
        out = window(self.p, self.doses, self.focuses, self.engine, lambda: self.stop, self.progress.emit)
        if out is None:
            return
        CD, LER = out
        E0, EL, dof = latitude(CD, self.doses, self.focuses, self.p.CD)
        self.finished_ok.emit(dict(doses=self.doses, focuses=self.focuses, CD=CD, LER=LER, E0=E0, EL=EL,
                                   DOF=dof, p=self.p, t=time.perf_counter() - t))


class Estimate(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)

    def __init__(self, p, n, conf, var, engine):
        super().__init__()
        self.args = p, n, conf, var, engine
        self.stop = False

    def run(self):
        t = time.perf_counter()
        p, n, conf, var, engine = self.args
        est = estimate(p, n, conf, var, engine, lambda: self.stop, self.progress.emit)
        if est is not None:
            est["t"] = time.perf_counter() - t
            self.finished_ok.emit(est)


class Log:
    def __init__(self, view):
        self.view = view
        LOGS.mkdir(exist_ok=True)
        self.file = LOGS / f"журнал_{datetime.now():%Y-%m-%d}.txt"
        self.hist = LOGS / "история_расчётов.csv"

    def write(self, text):
        text = re.sub(r"(?<![\w\\/.])-(?=\d)", "−", re.sub(r"(?<=\d)\.(?=\d)", ",", text))
        stamp = datetime.now().strftime("%H:%M:%S")
        lines = text.splitlines()
        out = [f"{stamp}  {lines[0]}"] + [f"          {s}" for s in lines[1:]]
        self.view.appendPlainText("\n".join(out))
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%d.%m.%Y} " + "\n".join(out) + "\n")

    def history(self, r):
        try:
            self._history(r)
        except OSError as e:
            self.write(f"Не удалось записать историю расчётов ({type(e).__name__}: {e}); расчёт при этом не затронут")

    def _history(self, r):
        p = asdict(r["p"])
        row = {"Время": f"{datetime.now():%d.%m.%Y %H:%M:%S}", **{PARAM_NAMES.get(k, k): v for k, v in p.items()},
               "Вычислительное ядро": r["engine"]}
        row.update({name + (f", {u}" if u else ""): f"{float(fn(r)):.4f}".replace(".", ",") for _, name, u, fn, _ in RESULTS})
        row = {k: (f"{v:g}".replace(".", ",") if isinstance(v, float) else v) for k, v in row.items()}
        fresh = not self.hist.exists()
        if not fresh:
            with open(self.hist, encoding="utf-8-sig") as f:
                head = f.readline().rstrip("\r\n").split(";")
            if head != list(row):
                self._archive_history()
                fresh = True
        mode = "w" if fresh else "a"
        with open(self.hist, mode, newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(row), delimiter=";")
            if fresh:
                w.writeheader()
            w.writerow(row)

    def _archive_history(self):
        old = self.hist.with_name(f"история_расчётов_до_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
        for attempt in range(10):
            try:
                self.hist.rename(old)
                break
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        else:
            old.write_bytes(self.hist.read_bytes())
        self.write(f"Набор столбцов истории изменился; прежний файл сохранён как {old.name}")


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP} (193/248 нм)")
        self.base = Process()
        self.rows, self.rid, self.cur, self.count = {}, 0, None, 0
        self.cache = OrderedDict()
        self.snaps, self.pw, self.sweep, self.est, self.est_job = [], None, None, None, None
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(1)
        self.sig = Signals()
        self.sig.done.connect(self.on_done)
        self.sig.failed.connect(self.on_fail)
        self.timer = QTimer(singleShot=True, interval=80, timeout=self.calc)

        self.build_params()
        self.build_plots()
        self.build_results()
        self.build_log()
        self.build_actions()
        self.log = Log(self.logview)
        self.progress = QProgressBar(maximumWidth=220, textVisible=False)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.engine_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.engine_lbl)

        self.settings = QSettings(str(BASE / "ler_sim.ini"), QSettings.IniFormat)
        self.resize(1600, 1000)
        if self.settings.value("geometry2"):
            self.restoreGeometry(self.settings.value("geometry2"))
            self.restoreState(self.settings.value("state2"))
        self.log.write(f"Запуск программы, версия {VERSION}. Вычислительное ядро: "
                       + ("C++ (OpenMP)" if _lercore else "Python (модуль C++ не найден)")
                       + f"\nЖурнал: {self.log.file}\nИстория расчётов: {self.log.hist}")
        manual = self.settings.value("threads", 0, int)
        self.statusBar().showMessage("Анализ компьютера…")
        QApplication.processEvents()
        self.hw = hw.analyze(manual or None)
        for w in (self.thr, self.thr_auto):
            w.blockSignals(True)
        self.thr.setValue(self.hw.threads)
        self.thr_auto.setChecked(not manual)
        self.thr.setEnabled(bool(manual))
        for w in (self.thr, self.thr_auto):
            w.blockSignals(False)
        self.log_machine()
        if self.hw.slow:
            self.a_auto.setChecked(False)
            self.log.write("Одна реализация считается дольше 1 с: автоматический пересчёт при движении ползунков "
                           "отключён, расчёт — по кнопке «Рассчитать» (F5). Включить: «Расчёт → Пересчитывать автоматически»")
        self.calc()

    def log_machine(self):
        m = self.hw
        txt = (f"Анализ компьютера: логических ядер {m.logical} (физических {m.physical}), свободно памяти "
               f"{m.mem_free:.1f} ГБ; потоков расчёта {m.threads} ({m.reason}). Одна реализация длиной 1,2 мкм — "
               f"{m.t_single:.2f} с; при пакетном расчёте — {m.t_task:.3f} с на реализацию")
        self.log.write(txt)
        c = lambda v, d: f"{v:.{d}f}".replace(".", ",")
        self.hw_lbl.setText(f"Логических ядер {m.logical} (физических {m.physical}), свободно памяти {c(m.mem_free, 1)} ГБ. "
                            f"Потоков расчёта {m.threads} ({m.reason}). Одна реализация длиной 1,2 мкм — "
                            f"{c(m.t_single, 2)} с, в пакете — {c(m.t_task, 3)} с на реализацию.")

    def set_threads(self):
        manual = 0 if self.thr_auto.isChecked() else self.thr.value()
        self.settings.setValue("threads", manual)
        self.hw = hw.analyze(manual or None)
        self.thr.blockSignals(True)
        self.thr.setValue(self.hw.threads)
        self.thr.blockSignals(False)
        self.log_machine()

    def expect(self, n, L):
        t = self.hw.estimate(n) * L / 1200.0
        return f"ожидаемое время ≈ {t:.0f} с" if t >= 2 else "ожидаемое время меньше 2 с"

    def build_params(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        for title, spec in GROUPS:
            g = QGroupBox(title)
            gv = QVBoxLayout(g)
            gv.setSpacing(0)
            if title == "Экспонирование":
                h = QHBoxLayout()
                lab = QLabel("Длина волны λ")
                lab.setFixedWidth(150)
                self.wl = QComboBox()
                self.wl.addItems(["193 нм (ArF)", "248 нм (KrF)"])
                self.wl.currentIndexChanged.connect(self.changed)
                h.addWidget(lab)
                h.addWidget(self.wl, 1)
                gv.addLayout(h)
            for key, *a in spec:
                gv.addWidget(self.row(key, *a))
            v.addWidget(g)
        v.addStretch()
        w2 = QWidget()
        v2 = QVBoxLayout(w2)
        v2.setContentsMargins(6, 6, 6, 6)
        g = QGroupBox("Модельный резист")
        gl = QGridLayout(g)
        gl.setVerticalSpacing(4)
        self.res_cells = {}
        for i, (key, label, unit, dec) in enumerate(RESIST):
            gl.addWidget(QLabel(label), i, 0)
            val = QLabel(alignment=Qt.AlignRight | Qt.AlignVCenter)
            gl.addWidget(val, i, 1)
            gl.addWidget(QLabel(unit), i, 2)
            self.res_cells[key] = (val, dec)
        note = QLabel("Свойства не калиброваны по измерениям. Калиброванный набор загружается из файла "
                      "параметров моделирования (Файл → Загрузить, Ctrl+O).", wordWrap=True)
        note.setStyleSheet("color: #555;")
        gl.addWidget(note, len(RESIST), 0, 1, 3)
        v2.addWidget(g)
        v2.addStretch()
        self.build_numerics()
        self.show_resist()
        tabs = QTabWidget(documentMode=True)
        for page, title in ((w, "Процесс"), (w2, "Резист")):
            sc = QScrollArea(widgetResizable=True, frameShape=QScrollArea.NoFrame)
            sc.setWidget(page)
            tabs.addTab(sc, title)
        reset = QPushButton("Вернуть исходные значения")
        reset.clicked.connect(self.reset)
        box = QWidget()
        bv = QVBoxLayout(box)
        bv.setContentsMargins(0, 0, 0, 6)
        bv.addWidget(tabs)
        bv.addWidget(reset)
        box.setMinimumWidth(440)
        d = QDockWidget("Параметры", self, objectName="params")
        d.setWidget(box)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)

    def build_numerics(self):
        self.num_dlg = QDialog(self)
        self.num_dlg.setWindowTitle("Параметры численного расчёта")
        gl = QGridLayout(self.num_dlg)
        self.L = QComboBox()
        self.L.addItems(["0,6 мкм", "1,2 мкм", "2,4 мкм", "4,8 мкм"])
        self.L.setCurrentIndex(1)
        self.nreal = QSpinBox(minimum=1, maximum=256, value=1)
        self.seed = QSpinBox(minimum=0, maximum=999999, value=self.base.seed)
        self.engine = QComboBox()
        self.engine.addItems(["C++", "Python (эталонная реализация)"])
        if _lercore is None:
            self.engine.setCurrentIndex(1)
            self.engine.model().item(0).setEnabled(False)
        tips = ["Длина рассчитываемого фрагмента линии; нижняя пространственная частота спектра 1/L",
                "Число независимых реализаций (метод Монте-Карло); метрики усредняются по ансамблю",
                "Начальное значение генератора случайных чисел; одинаковое значение даёт одинаковый результат",
                "Обе реализации решают одну задачу; эталонная медленнее и служит для проверки"]
        for i, (t, wdg) in enumerate((("Длина фрагмента линии L", self.L), ("Число реализаций N", self.nreal),
                                      ("Начальное значение генератора", self.seed), ("Вычислительное ядро", self.engine))):
            lab = QLabel(t)
            lab.setToolTip(tips[i])
            wdg.setToolTip(tips[i])
            gl.addWidget(lab, i, 0)
            gl.addWidget(wdg, i, 1)
        self.thr_auto = QCheckBox("автоматически")
        self.thr = QSpinBox(minimum=1, maximum=os.cpu_count() or 1)
        tip = ("Число потоков влияет только на скорость: результат побитово одинаков при любом числе потоков. "
               "Автоматически — все логические ядра, кроме одного для интерфейса, с ограничением по свободной памяти")
        h = QHBoxLayout()
        h.addWidget(self.thr)
        h.addWidget(self.thr_auto)
        lab = QLabel("Потоков расчёта")
        for w in (lab, self.thr, self.thr_auto):
            w.setToolTip(tip)
        gl.addWidget(lab, 4, 0)
        gl.addLayout(h, 4, 1)
        self.hw_lbl = QLabel(wordWrap=True)
        self.hw_lbl.setStyleSheet("color: #555;")
        gl.addWidget(self.hw_lbl, 5, 0, 1, 2)
        grid = QLabel(f"Сетка Δx × Δy = {self.base.dx:g} × {self.base.dy:g} нм, шагов PEB: 30. Сходимость по сетке "
                      "и шагу проверена тестами (см. руководство, раздел «Проверка модели»).", wordWrap=True)
        grid.setStyleSheet("color: #555;")
        gl.addWidget(grid, 6, 0, 1, 2)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.num_dlg.close)
        gl.addWidget(close, 7, 1)
        self.thr_auto.toggled.connect(lambda on: (self.thr.setEnabled(not on), self.set_threads()))
        self.thr.valueChanged.connect(lambda _: self.set_threads())
        for s in (self.L.currentIndexChanged, self.nreal.valueChanged, self.seed.valueChanged,
                  self.engine.currentIndexChanged):
            s.connect(self.changed)

    def show_resist(self):
        for key, (val, dec) in self.res_cells.items():
            val.setText(f"{getattr(self.base, key):.{dec}f}".replace(".", ","))

    def row(self, key, label, unit, lo, hi, step, dec, tip):
        r = Row(label, unit, lo, hi, step, dec, getattr(self.base, key), tip)
        r.changed.connect(self.changed)
        self.rows[key] = r
        return r

    def build_plots(self):
        self.tabs = QTabWidget(documentMode=True)
        self.overlay = QCheckBox("Показать опорный режим")
        self.overlay.setEnabled(False)
        self.overlay.toggled.connect(self.redraw)
        self.p_contour, self.p_sec, self.p_psd, self.p_ens = Plot(self.overlay), Plot(), Plot(), Plot()
        for p, t in ((self.p_contour, "Контур линии"), (self.p_sec, "Поперечное сечение"),
                     (self.p_psd, "Спектральная плотность шероховатости"), (self.p_ens, "Статистика реализаций")):
            self.tabs.addTab(p, t)

        pw = QWidget()
        pv = QVBoxLayout(pw)
        pv.setContentsMargins(4, 4, 4, 0)
        bar = QHBoxLayout()
        self.d0 = QDoubleSpinBox(minimum=5, maximum=100, value=24, decimals=1, singleStep=1)
        self.d1 = QDoubleSpinBox(minimum=5, maximum=100, value=42, decimals=1, singleStep=1)
        self.dn = QSpinBox(minimum=3, maximum=31, value=10)
        self.f0 = QSpinBox(minimum=-1000, maximum=0, value=-300, singleStep=25)
        self.f1 = QSpinBox(minimum=0, maximum=1000, value=300, singleStep=25)
        self.fn = QSpinBox(minimum=3, maximum=61, value=13)
        self.pw_btn = QPushButton("Рассчитать окно процесса")
        self.pw_btn.clicked.connect(self.run_sweep)
        for t, w in (("E от", self.d0), ("до", self.d1), ("мДж/см², точек", self.dn), ("   Δz от", self.f0),
                     ("до", self.f1), ("нм, точек", self.fn)):
            bar.addWidget(QLabel(t))
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(self.pw_btn)
        pv.addLayout(bar)
        self.pw_tabs = QTabWidget(tabPosition=QTabWidget.South, documentMode=True)
        self.p_boss, self.p_map = Plot(), Plot()
        self.pw_tabs.addTab(self.p_boss, "Кривые Боссунга")
        self.pw_tabs.addTab(self.p_map, "Окно процесса и энергетическая широта")
        pv.addWidget(self.pw_tabs)
        for p in (self.p_boss, self.p_map):
            figs._empty(p.fig, "Задайте диапазоны и нажмите «Рассчитать окно процесса»")
        self.tabs.addTab(pw, "Окно процесса")
        self.tabs.addTab(self.build_stats(), "Статистическая оценка")
        self.tabs.currentChanged.connect(lambda _: self.redraw())

        c = QWidget()
        cv = QVBoxLayout(c)
        cv.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel(objectName="summary")
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cv.addWidget(self.summary)
        self.warn = QLabel(objectName="warn", wordWrap=True)
        self.warn.hide()
        cv.addWidget(self.warn)
        cv.addWidget(self.tabs)
        self.setCentralWidget(c)

    def build_stats(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 0)
        bar = QHBoxLayout()
        self.st_n = QSpinBox(minimum=8, maximum=512, value=32, singleStep=8)
        self.st_conf = QComboBox()
        self.st_conf.addItems(["0,90", "0,95", "0,99"])
        self.st_conf.setCurrentIndex(1)
        self.st_var = {}
        for t, wdg, tip in (("Число реализаций N", self.st_n, "Чем больше N, тем уже доверительный интервал среднего"),
                            ("Доверительная вероятность P", self.st_conf, "Вероятность, с которой величина попадает в интервал")):
            lab = QLabel(t)
            lab.setToolTip(tip)
            wdg.setToolTip(tip)
            bar.addWidget(lab)
            bar.addWidget(wdg)
            bar.addSpacing(10)
        bar.addWidget(QLabel("   Нестабильность процесса (СКО):"))
        for key, t, unit, hi, step, tip in (
                ("dose", "σ<sub>E</sub>", "%", 10, 0.1, "Случайное отклонение дозы от партии к партии, в процентах от заданной"),
                ("focus", "σ<sub>Δz</sub>", "нм", 100, 1, "Случайная расфокусировка: неплоскостность пластины, точность автофокуса"),
                ("T_peb", "σ<sub>T</sub>", "°C", 3, 0.05, "Неоднородность температуры плиты PEB"),
                ("L_D", "σ<sub>L<sub>D</sub></sub>", "%", 20, 0.5, "Разброс длины диффузии кислоты от партии резиста")):
            sb = QDoubleSpinBox(minimum=0, maximum=hi, singleStep=step, decimals=2, value=0)
            sb.setToolTip(tip)
            lab = QLabel(t)
            lab.setToolTip(tip)
            bar.addWidget(lab)
            bar.addWidget(sb)
            bar.addWidget(QLabel(unit))
            self.st_var[key] = sb
        bar.addStretch()
        self.st_btn = QPushButton("Выполнить оценку")
        self.st_btn.clicked.connect(self.run_estimate)
        bar.addWidget(self.st_btn)
        v.addLayout(bar)
        heads = ["Величина", "Ед.", "Среднее x̄", "СКО s", "ДИ среднего", "Интервал предсказания",
                 "СКО без нестабильности процесса", "Вклад нестабильности процесса, %"]
        self.st_table = table(4, heads)
        hh = self.st_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        self.st_table.setFixedHeight(28 + 22 * 4)
        v.addWidget(self.st_table)
        self.st_note = QLabel(wordWrap=True)
        self.st_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.st_note)
        self.p_stat = Plot()
        figs._empty(self.p_stat.fig, "Задайте число реализаций и нестабильность процесса и нажмите «Выполнить оценку»")
        v.addWidget(self.p_stat, 1)
        return w

    def run_estimate(self):
        if self.est_job is not None:
            self.est_job.stop = True
            return
        p, n = self.params(), self.st_n.value()
        conf = (0.90, 0.95, 0.99)[self.st_conf.currentIndex()]
        var = {k: sb.value() for k, sb in self.st_var.items() if sb.value() > 0}
        self.est_job = Estimate(p, n, conf, var, self.engine_name())
        self.est_job.progress.connect(self.on_estimate_progress)
        self.est_job.finished_ok.connect(self.on_estimate)
        self.est_job.finished.connect(self.on_estimate_end)
        self.st_btn.setText("Остановить")
        self.progress.show()
        names = dict(dose="доза, %", focus="фокус, нм", T_peb="T PEB, °C", L_D="длина диффузии, %")
        cost = n if not var else 2 * n * self.hw.t_single / self.hw.t_task
        self.log.write((f"Статистическая оценка ({self.expect(cost, p.L)}): N = {n}, P = {conf:.2f}; "
                        "нестабильность процесса (СКО): "
                        + (", ".join(f"{names[k]} {v:g}" for k, v in var.items()) if var else "не задана"))
                       .replace(".", ","))
        self.est_job.start()

    def on_estimate_progress(self, k, n):
        self.progress.setMaximum(n)
        self.progress.setValue(k)
        self.statusBar().showMessage(f"Статистическая оценка: выполнено {k} расчётов из {n}")

    def on_estimate(self, est):
        self.est = est
        fmt = lambda v, d=2: f"{v:.{d}f}".replace(".", ",").replace("-", "−")
        iv = lambda a, d=2: f"{fmt(a[0], d)} … {fmt(a[1], d)}"
        lines = []
        for i, (key, q) in enumerate(est["quantities"].items()):
            s = q["total"]
            cells = [q["name"], q["unit"], fmt(s["mean"]), fmt(s["sd"]), iv(s["ci"]), iv(s["pi"]),
                     fmt(q["stoch"]["sd"]) if est["var"] else "—",
                     fmt(100 * q["process_share"], 0) if est["var"] else "—"]
            for j, c in enumerate(cells):
                it = QTableWidgetItem(c)
                if j > 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.st_table.setItem(i, j, it)
            lines.append(f"{q['name']}: среднее {fmt(s['mean'])} {q['unit']}, СКО {fmt(s['sd'])}, ДИ среднего {iv(s['ci'])}, "
                         f"интервал предсказания {iv(s['pi'])}"
                         + (f", вклад нестабильности процесса {fmt(100 * q['process_share'], 0)} %" if est["var"] else ""))
        ps, pc = est["point_stats"], f"{100 * est['conf']:g}".replace(".", ",")
        note = (f"Отклонения точек края от номинала ({est['points'].size} точек): среднее {fmt(ps['mean'])} нм, "
                f"СКО {fmt(ps['sd'])} нм, среднее абсолютное отклонение {fmt(ps['mad'])} нм; "
                f"интервал x̄ ± zσ при P = {pc} %: {iv(ps['band'])} нм (в нём {fmt(100 * ps['inside'], 1)} % точек), "
                f"по выборочным квантилям: {iv(ps['q'])} нм. Время расчёта {fmt(est['t'], 1)} с.")
        self.st_note.setText(note)
        figs.stats(self.p_stat.fig, est)
        self.p_stat.draw()
        self.log.write("Статистическая оценка готова\n" + "\n".join(lines) + "\n" + note)

    def on_estimate_end(self):
        if self.est_job.stop:
            self.log.write("Статистическая оценка остановлена")
        self.est_job = None
        self.st_btn.setText("Выполнить оценку")
        self.progress.hide()
        self.statusBar().clearMessage()

    def build_results(self):
        names = {k: (n, u) for k, n, u, _, _ in RESULTS}
        names.update(pw0=("Доза номинального размера E₀ (Δz = 0)", "мДж/см²"), pw1=("Энергетическая широта EL (Δz = 0)", "%"),
                     pw2=("Глубина фокуса DOF (EL ≥ 5 %)", "нм"), pw3=("Время расчёта окна процесса", "с"))
        layout = [("Геометрия и шероховатость линии", ["cd", "dcd", "lerl", "lerr", "lwr", "rho", "lost", "n", "t"]),
                  ("Изображение, спектральные параметры, окно процесса",
                   ["nils", "ldiff", "fs", "fxi", "fa", "fn", "pw0", "pw1", "pw2", "pw3"])]
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 4, 4, 4)
        self.tables, self.cells = [], {}
        for title, keys in layout:
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            bl.setContentsMargins(4, 4, 4, 4)
            t = table(len(keys), ["Показатель", "Ед.", "Текущий"])
            for i, k in enumerate(keys):
                n, u = names[k]
                it = QTableWidgetItem(n)
                it.setToolTip(n)
                t.setItem(i, 0, it)
                t.setItem(i, 1, QTableWidgetItem(u))
                self.cells[k] = (t, i)
            hh = t.horizontalHeader()
            hh.setMinimumSectionSize(70)
            hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            hh.setStretchLastSection(False)
            t.setMinimumHeight(26 + 22 * len(keys))
            bl.addWidget(t)
            h.addWidget(box, 1)
            self.tables.append(t)
        side = QVBoxLayout()
        for text, fn, tip in (("Запомнить режим", self.snapshot, "Добавить текущие результаты отдельным столбцом"),
                              ("Убрать последний", self.drop_snapshot, "Удалить последний запомненный столбец"),
                              ("Копировать", self.copy_table, "Скопировать обе таблицы для вставки в Excel"),
                              ("Сохранить в CSV…", self.export_table, "Сохранить обе таблицы в файл CSV")):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(fn)
            side.addWidget(btn)
        side.addStretch()
        h.addLayout(side)
        self.d_res = QDockWidget("Результаты", self, objectName="results")
        self.d_res.setWidget(w)
        self.d_res.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.d_res.setTitleBarWidget(QWidget())
        self.addDockWidget(Qt.BottomDockWidgetArea, self.d_res)

    def build_log(self):
        self.logview = QPlainTextEdit(readOnly=True)
        self.logview.setMaximumBlockCount(5000)
        self.logview.setLineWrapMode(QPlainTextEdit.NoWrap)
        d = QDockWidget("Журнал расчётов", self, objectName="log")
        d.setWidget(self.logview)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)
        d.setTitleBarWidget(QWidget())
        self.tabifyDockWidget(self.d_res, d)
        self.d_res.raise_()
        self.setTabPosition(Qt.BottomDockWidgetArea, QTabWidget.North)
        self.resizeDocks([d], [330], Qt.Vertical)

    def build_actions(self):
        def act(text, fn, key=None, checkable=False, checked=False):
            a = QAction(text, self, checkable=checkable)
            if checkable:
                a.setChecked(checked)
            if key:
                a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
            return a
        self.a_calc = act("Рассчитать", self.calc, "F5")
        self.a_auto = act("Пересчитывать автоматически", lambda: None, checkable=True, checked=True)
        self.a_pw = act("Окно процесса", self.goto_sweep, "F6")
        self.a_st = act("Статистическая оценка", lambda: self.tabs.setCurrentIndex(5), "F7")
        self.a_snap = act("Запомнить режим", self.snapshot, "Ctrl+M")
        self.a_num = act("Параметры численного расчёта…", lambda: (self.num_dlg.show(), self.num_dlg.raise_()), "Ctrl+P")
        a_open = act("Загрузить параметры моделирования…", self.load_params, "Ctrl+O")
        a_save = act("Сохранить параметры моделирования…", self.save_params, "Ctrl+S")
        a_res = act("Экспорт таблицы результатов (CSV)…", self.export_table)
        a_edges = act("Экспорт контуров краёв (CSV)…", self.export_edges)
        a_figs = act("Сохранить все графики…", self.export_figs)
        a_logs = act("Открыть папку журнала", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOGS))))
        a_clear = act("Очистить окно журнала", lambda: self.logview.clear())

        tb = self.addToolBar("Основное")
        tb.setObjectName("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for a in (self.a_calc, self.a_auto, None, self.a_pw, self.a_st, self.a_snap, None, a_figs):
            tb.addSeparator() if a is None else tb.addAction(a)

        mb = self.menuBar()
        m = mb.addMenu("Файл")
        for a in (a_open, a_save, None, a_res, a_edges, a_figs, None, a_logs, None):
            m.addSeparator() if a is None else m.addAction(a)
        m.addAction(act("Выход", self.close, "Ctrl+Q"))
        m = mb.addMenu("Расчёт")
        for a in (self.a_calc, self.a_auto, self.a_pw, self.a_st, self.a_snap, None, self.a_num):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("Вид")
        for d in self.findChildren(QDockWidget):
            m.addAction(d.toggleViewAction())
        m.addAction(a_clear)
        m = mb.addMenu("Справка")
        m.addAction(act("Руководство пользователя", self.open_guide, "F1"))
        m.addAction(act("Кратко о модели расчёта", self.about_model))
        m.addAction(act("О программе", lambda: QMessageBox.about(self, "О программе",
            f"{APP}\nВерсия {VERSION}\n\nВычислительное ядро: "
            + (f"C++ (OpenMP), потоков: {_lercore.threads()}" if _lercore else "Python / NumPy")
            + f"\nПапка программы: {BASE}")))

    def changed(self):
        if self.a_auto.isChecked():
            self.timer.start()

    def params(self):
        v = {k: r.value() for k, r in self.rows.items()}
        v["pitch"] = max(v["pitch"], v["CD"] + 40)
        return replace(self.base, **v, wl=(193.0, 248.0)[self.wl.currentIndex()],
                       L=(600.0, 1200.0, 2400.0, 4800.0)[self.L.currentIndex()], seed=self.seed.value())

    def engine_name(self):
        return "cpp" if self.engine.currentIndex() == 0 and _lercore else "numpy"

    def calc(self):
        self.rid += 1
        p, n, eng = self.params(), self.nreal.value(), self.engine_name()
        key = (tuple(asdict(p).values()), n, eng)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.on_done(self.rid, self.cache[key], cached=True)
            return
        self.statusBar().showMessage("Идёт расчёт…")
        self.pool.start(Job(self.rid, p, n, eng, self.sig))

    def on_fail(self, rid, msg):
        self.log.write(f"Ошибка расчёта: {msg}")
        self.statusBar().showMessage("Ошибка расчёта, подробности в журнале")

    def on_done(self, rid, r, cached=False):
        if not cached:
            key = (tuple(asdict(r["p"]).values()), r["n"], "cpp" if r["engine"] == "C++" else "numpy")
            self.cache[key] = r
            while sum(v["mb"] for v in self.cache.values()) > CACHE_MB and len(self.cache) > 1:
                self.cache.popitem(last=False)
        if rid != self.rid:
            return
        self.cur = r
        self.count += 1
        self.redraw()
        self.fill(2, r)
        p, m = r["p"], r["m"]
        warn = []
        if r["lost_frac"] > 0:
            warn.append(f"Линия разрывается: проявитель дошёл до центра линии в {100*r['lost_frac']:.1f} % сечений.")
        if r["L_diff"] < 2 * p.dy:
            warn.append(f"Фактическая длина диффузии {r['L_diff']:.1f} нм меньше двух шагов сетки ({2 * p.dy:g} нм): "
                        "размытие кислоты сеткой не разрешается, шероховатость завышена.")
        if r["NILS"] < 1.5:
            warn.append(f"Низкий контраст изображения на краю (NILS = {r['NILS']:.2f}): результат ненадёжен.")
        if p.pitch != self.rows["pitch"].value():
            warn.append(f"Шаг увеличен до {p.pitch:.0f} нм: он должен быть больше ширины линии хотя бы на 40 нм.")
        self.summary.setText(
            (f"CD = <b>{m['CD']:.1f} нм</b> (ΔCD = {m['CD'] - p.CD:+.1f} нм)&nbsp;&nbsp;&nbsp;&nbsp; "
             f"LER (3σ) = <b>{0.5 * (m['LER_L'] + m['LER_R']):.2f} нм</b>&nbsp;&nbsp;&nbsp;&nbsp; "
             f"LWR (3σ) = <b>{m['LWR']:.2f} нм</b>&nbsp;&nbsp;&nbsp;&nbsp; ρ = <b>{m['rho']:+.2f}</b>&nbsp;&nbsp;&nbsp;&nbsp; "
             f"NILS = <b>{r['NILS']:.2f}</b>")
            .replace(".", ",").replace("-", "−"))
        self.warn.setText("  ".join(warn))
        self.warn.setVisible(bool(warn))
        self.engine_lbl.setText(f"Ядро: {r['engine']}")
        ler = 0.5 * (m["LER_L"] + m["LER_R"])
        if cached:
            self.statusBar().showMessage("Результат взят из кэша", 5000)
            self.log.write(f"Расчёт № {self.count}: режим уже считался, результат взят из кэша без пересчёта "
                           f"(доза {p.dose:.1f} мДж/см², фокус {p.focus:+.0f} нм, PEB {p.T_peb:.0f} °C / {p.t_peb:.0f} с, "
                           f"проявление {p.t_dev:.0f} с): ширина {m['CD']:.2f} нм, шероховатость края {ler:.2f} нм")
            return
        self.statusBar().showMessage(f"Готово за {r['t_total']:.2f} с".replace(".", ","), 8000)
        lines = [f"Расчёт № {self.count}",
                 f"Режим: λ {p.wl:.0f} нм, NA {p.NA:.2f}, σ {p.sigma_s:.2f}; линия {p.CD:.0f} нм, шаг {p.pitch:.0f} нм; "
                 f"доза {p.dose:.1f} мДж/см², фокус {p.focus:+.0f} нм; PEB {p.T_peb:.0f} °C, {p.t_peb:.0f} с; "
                 f"проявление {p.t_dev:.0f} с; участок {p.L/1e3:.1f} мкм, повторов {r['n']}, генератор {p.seed}"]
        lines += [f"{name} ({dt:.3f} с): {text}" for name, text, dt in r["stages"]]
        s, xi, a, _, fl = r["fit"]
        lines.append(f"Аппроксимация спектра: СКО коррелированной части {s:.2f} нм, длина корреляции {xi:.1f} нм, "
                     f"показатель шероховатости {a:.2f}, уровень белого шума {fl:.2f} нм³")
        lines.append(f"Итог: ширина {m['CD']:.2f} нм (отклонение {m['CD'] - p.CD:+.2f}), шероховатость края "
                     f"{ler:.2f} нм, ширины {m['LWR']:.2f} нм (3σ); всего {r['t_total']:.2f} с")
        self.log.write("\n".join(lines))
        self.log.history(r)

    def redraw(self):
        if self.cur is None:
            return
        r, p = self.cur, self.cur["p"]
        i = self.tabs.currentIndex()
        ref = self.snaps[-1][1] if self.snaps and self.overlay.isChecked() else None
        if i == 0:
            figs.contour(self.p_contour.fig, r, p, ref)
            self.p_contour.draw()
        elif i == 1:
            figs.section(self.p_sec.fig, r, p)
            self.p_sec.draw()
        elif i == 2:
            figs.psd(self.p_psd.fig, r, r["fit"])
            self.p_psd.draw()
        elif i == 3:
            figs.ensemble(self.p_ens.fig, r)
            self.p_ens.draw()

    def put(self, key, col, text):
        t, i = self.cells[key]
        if t.columnCount() <= col:
            t.setColumnCount(col + 1)
        it = QTableWidgetItem(text)
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        t.setItem(i, col, it)

    def fill(self, col, r):
        for k, _, _, fn, dec in RESULTS:
            self.put(k, col, f"{fn(r):.{dec}f}".replace(".", ",").replace("-", "−"))
        if self.pw is not None and col == 2:
            self.fill_pw()

    def fill_pw(self):
        pw = self.pw
        j = int(np.argmin(np.abs(pw["focuses"])))
        for i, v in enumerate((pw["E0"][j], pw["EL"][j], pw["DOF"], pw["t"])):
            self.put(f"pw{i}", 2, "—" if not np.isfinite(v) else f"{v:.1f}".replace(".", ","))

    def snapshot(self):
        if self.cur is None:
            return
        n = len(self.snaps) + 1
        self.snaps.append((f"Режим {n}", self.cur))
        col = 2 + n
        self.fill(col, self.cur)
        p = self.cur["p"]
        for t in self.tables:
            t.setColumnCount(max(t.columnCount(), col + 1))
            hdr = QTableWidgetItem(f"Режим {n}")
            hdr.setToolTip(f"доза {p.dose:.1f} мДж/см², фокус {p.focus:+.0f} нм, PEB {p.T_peb:.0f} °C / {p.t_peb:.0f} с, "
                           f"проявление {p.t_dev:.0f} с, λ {p.wl:.0f} нм, NA {p.NA:.2f}")
            t.setHorizontalHeaderItem(col, hdr)
        self.overlay.setEnabled(True)
        self.log.write(f"Текущие результаты запомнены как «Режим {n}»")

    def drop_snapshot(self):
        if self.snaps:
            self.snaps.pop()
            for t in self.tables:
                t.setColumnCount(3 + len(self.snaps))
            self.overlay.setEnabled(bool(self.snaps))
            self.redraw()

    def table_rows(self):
        out = []
        for t in self.tables:
            cols = t.columnCount()
            if not out:
                out.append([t.horizontalHeaderItem(c).text() for c in range(cols)])
            out += [[(t.item(i, c).text() if t.item(i, c) else "") for c in range(cols)] for i in range(t.rowCount())]
        return out

    def copy_table(self):
        QGuiApplication.clipboard().setText("\n".join("\t".join(r) for r in self.table_rows()))
        self.statusBar().showMessage("Таблица скопирована; её можно вставить в Excel", 5000)

    def export_table(self):
        f, _ = QFileDialog.getSaveFileName(self, "Таблица результатов", str(BASE / "результаты.csv"), "CSV (*.csv)")
        if f:
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                csv.writer(fh, delimiter=";").writerows(self.table_rows())
            self.log.write(f"Таблица результатов сохранена: {f}")

    def export_edges(self):
        if self.cur is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Контуры краёв", str(BASE / "контуры.csv"), "CSV (*.csv)")
        if f:
            r = self.cur
            head = ["y, нм"] + [f"левый край {i+1}, нм" for i in range(len(r["xL"]))] + \
                   [f"правый край {i+1}, нм" for i in range(len(r["xR"]))]
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh, delimiter=";")
                w.writerow(head)
                for row in np.c_[r["y"], r["xL"].T, r["xR"].T]:
                    w.writerow([f"{v:.4f}".replace(".", ",") for v in row])
            self.log.write(f"Контуры краёв сохранены: {f}")

    def export_figs(self):
        if self.cur is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Папка для графиков", str(BASE))
        if not d:
            return
        r, p = self.cur, self.cur["p"]
        out = [("контур", lambda f: figs.contour(f, r, p)), ("сечение", lambda f: figs.section(f, r, p)),
               ("спектр", lambda f: figs.psd(f, r, r["fit"])), ("повторы", lambda f: figs.ensemble(f, r))]
        if self.est is not None:
            out.append(("статистическая_оценка", lambda f: figs.stats(f, self.est)))
        if self.pw is not None:
            out += [("ширина_от_фокуса", lambda f: figs.bossung(f, self.pw, self.pw["p"])),
                    ("окно_процесса", lambda f: figs.window(f, self.pw, self.pw["p"]))]
        for name, draw in out:
            fig = Figure(figsize=(7.2, 3.6), layout="constrained")
            draw(fig)
            fig.savefig(f"{d}/{name}.png", dpi=600)
            fig.savefig(f"{d}/{name}.pdf")
        self.log.write(f"Графики сохранены (PNG 600 dpi и PDF): {d}")

    def save_params(self):
        f, _ = QFileDialog.getSaveFileName(self, "Сохранить параметры моделирования", str(BASE / "параметры.json"),
                                           "Параметры моделирования (*.json)")
        if f:
            data = dict(программа=APP, версия=VERSION, параметры=asdict(self.params()),
                        повторов=self.nreal.value(), ядро=self.engine_name())
            Path(f).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log.write(f"Параметры моделирования сохранены: {f}")

    def load_params(self):
        f, _ = QFileDialog.getOpenFileName(self, "Загрузить параметры моделирования", str(BASE),
                                           "Параметры моделирования (*.json)")
        if not f:
            return
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            self.apply(data["параметры"], data.get("повторов", 1))
            self.log.write(f"Параметры моделирования загружены: {f}")
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, "Параметры моделирования", f"Не удалось прочитать файл:\n{e}")

    def apply(self, prm, n=1):
        prm = asdict(Process.from_dict({**asdict(self.base), **prm}))
        auto = self.a_auto.isChecked()
        self.a_auto.setChecked(False)
        known = {f.name for f in fields(Process)}
        for k, v in prm.items():
            if k in self.rows and not self.rows[k].spin.minimum() <= v <= self.rows[k].spin.maximum():
                self.log.write(f"Значение «{PARAM_NAMES.get(k, k)}» = {v:g} из файла вне допустимого диапазона "
                               f"{self.rows[k].spin.minimum():g}…{self.rows[k].spin.maximum():g}; "
                               f"использовано ближайшее допустимое")
            if k in self.rows:
                self.rows[k].set(v)
        if "wl" in prm:
            self.wl.setCurrentIndex(0 if prm["wl"] < 220 else 1)
        if "L" in prm:
            self.L.setCurrentIndex(int(np.argmin(np.abs(np.array([600, 1200, 2400, 4800]) - prm["L"]))))
        if "seed" in prm:
            self.seed.setValue(int(prm["seed"]))
        self.nreal.setValue(int(n))
        self.base = replace(self.base, **{k: v for k, v in prm.items() if k in known and k not in self.rows})
        self.show_resist()
        self.a_auto.setChecked(auto)
        self.calc()

    def reset(self):
        self.base = Process()
        self.apply(asdict(self.base), 1)
        self.log.write("Параметры возвращены к исходным значениям")

    def goto_sweep(self):
        self.tabs.setCurrentIndex(4)

    def run_sweep(self):
        if self.sweep is not None:
            self.sweep.stop = True
            return
        p = self.params()
        doses = np.linspace(self.d0.value(), self.d1.value(), self.dn.value())
        focuses = np.linspace(self.f0.value(), self.f1.value(), self.fn.value())
        self.sweep = Sweep(p, doses, focuses, self.engine_name())
        self.sweep.progress.connect(self.on_sweep_progress)
        self.sweep.finished_ok.connect(self.on_sweep)
        self.sweep.finished.connect(self.on_sweep_end)
        self.pw_btn.setText("Остановить")
        self.progress.show()
        self.log.write(f"Окно процесса ({self.expect(doses.size * focuses.size, p.L)}): "
                       f"доза {doses[0]:.1f}…{doses[-1]:.1f} мДж/см² ({doses.size} точек), "
                       f"фокус {focuses[0]:+.0f}…{focuses[-1]:+.0f} нм ({focuses.size} точек); полный стохастический "
                       f"расчёт в каждой точке с общими случайными числами, участок {p.L / 1e3:g} мкм".replace(".", ","))
        self.sweep.start()

    def on_sweep_progress(self, k, n):
        self.progress.setMaximum(n)
        self.progress.setValue(k)
        self.statusBar().showMessage(f"Окно процесса: точка {k} из {n}")

    def on_sweep(self, pw):
        self.pw = pw
        p = pw["p"]
        figs.bossung(self.p_boss.fig, pw, p)
        figs.window(self.p_map.fig, pw, p)
        self.p_boss.draw()
        self.p_map.draw()
        self.fill_pw()
        j = int(np.argmin(np.abs(pw["focuses"])))
        txt = (f"Окно процесса готово за {pw['t']:.1f} с: доза номинального размера E₀ (Δz = 0) — "
               f"{pw['E0'][j]:.2f} мДж/см², энергетическая широта EL (±10 % CD, Δz = 0) — {pw['EL'][j]:.1f} %, "
               f"глубина фокуса DOF (EL ≥ 5 %) — {pw['DOF']:.0f} нм")
        if np.isfinite(pw["LER"]).any():
            txt += f"\nШероховатость края LER (3σ) на сетке: от {np.nanmin(pw['LER']):.2f} до {np.nanmax(pw['LER']):.2f} нм"
        if pw["DOF"] >= pw["focuses"][-1] - pw["focuses"][0]:
            txt += "\nEL ≥ 5 % во всём диапазоне фокуса: глубина фокуса ограничена диапазоном расчёта, его нужно расширить"
        self.log.write(txt)

    def on_sweep_end(self):
        if self.sweep.stop:
            self.log.write("Расчёт окна процесса остановлен")
        self.sweep = None
        self.pw_btn.setText("Рассчитать окно процесса")
        self.progress.hide()
        self.statusBar().clearMessage()

    def open_guide(self):
        if not GUIDE.exists() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(GUIDE))):
            QMessageBox.warning(self, "Руководство пользователя", f"Не удалось открыть файл руководства:\n{GUIDE}")

    def about_model(self):
        QMessageBox.information(self, "Что считает программа", (
            "1. Изображение. Скалярное частично когерентное изображение решётки линий: числовая апертура, "
            "круглый источник, расфокусировка.\n\n"
            "2. Экспонирование. Резист (вид сверху) разбит на ячейки 1 × 1 нм; число молекул фотогенератора кислоты "
            "и тушителя в ячейке — по Пуассону, разложение каждой молекулы — с вероятностью 1 − exp(−C·E·I).\n\n"
            "3. Термообработка. Молекулы — облака радиуса реакции r_c; кислота диффундирует, мгновенно "
            "нейтрализуется неподвижным тушителем и снимает защиту со смолы; D и k_d — по закону Аррениуса.\n\n"
            "4. Проявление. Скорость растворения — модель Мака; фронт идёт от центров пробелов в плоскости x–y "
            "по уравнению эйконала и обходит медленные участки; край — положение фронта в момент t_dev.\n\n"
            "5. Обработка. Ширина, шероховатость краёв и ширины (3σ), корреляция краёв, спектр шероховатости "
            "и его аппроксимация.\n\n"
            "Окно процесса и статистическая оценка — тот же полный стохастический расчёт в каждой точке.\n\n"
            "Свойства резиста условные, не калиброваны по измерениям.\n\nПодробно — «Руководство пользователя» (F1)."))

    def closeEvent(self, e):
        for job in (self.sweep, self.est_job):
            if job is not None:
                job.stop = True
                job.wait(3000)
        self.pool.waitForDone(10000)
        self.settings.setValue("geometry2", self.saveGeometry())
        self.settings.setValue("state2", self.saveState())
        super().closeEvent(e)


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    QLocale.setDefault(QLocale(QLocale.Russian, QLocale.Russia))
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 9))
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
