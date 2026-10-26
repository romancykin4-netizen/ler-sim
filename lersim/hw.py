"""Анализ компьютера при запуске: ядра, свободная память, скорость расчёта; выбор числа потоков.

Число потоков влияет только на скорость: у каждой строки сетки и каждой реализации свой поток случайных
чисел, а проявление делится на фиксированное число полос, поэтому результат побитово одинаков при любом
числе потоков и на любом компьютере.
"""
import ctypes
import os
import sys
import time
from dataclasses import dataclass, replace
from .process import Process, simulate, _lercore

# Рабочая память одного потока на ячейку сетки: четыре поля float (кислота, её копия, тушитель, ∫[H]dt)
# и два поля double проявления (время прихода, медленность) — 32 байта; с запасом на кольцо и буферы — 40.
BYTES_PER_CELL = 40
# Наибольшая сетка, которую можно задать в программе: период 600 нм × фрагмент 4,8 мкм при шаге 1 нм.
MAX_CELLS = 600 * 4800


@dataclass
class Machine:
    logical: int
    physical: int
    mem_free: float          # ГБ
    threads: int
    t_single: float          # с: одна реализация 1,2 мкм, все потоки на ней
    t_task: float            # с: одна реализация 1,2 мкм при расчёте пачки, приведено к одной реализации
    reason: str

    def estimate(self, n):
        """Ожидаемое время n независимых реализаций длиной 1,2 мкм (окно процесса, статистическая оценка)."""
        return self.t_single if n <= 1 else n * self.t_task

    @property
    def slow(self):
        return self.t_single > 1.0


def physical_cores(logical):
    if sys.platform == "win32":
        # GetLogicalProcessorInformation: записи RelationProcessorCore (0) — физические ядра.
        class Info(ctypes.Structure):
            _fields_ = [("mask", ctypes.c_size_t), ("rel", ctypes.c_int), ("data", ctypes.c_byte * 16)]
        n = ctypes.c_ulong(0)
        k32 = ctypes.windll.kernel32
        k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
        buf = (Info * (n.value // ctypes.sizeof(Info)))()
        if k32.GetLogicalProcessorInformation(buf, ctypes.byref(n)):
            return max(1, sum(1 for e in buf if e.rel == 0))
        return logical
    try:
        cores = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    cores.add((phys, core))
        return len(cores) or logical
    except OSError:
        return logical


def free_memory_gb():
    if sys.platform == "win32":
        class Mem(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
        m = Mem()
        m.len = ctypes.sizeof(Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 2**30
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError):
        return 4.0


def choose_threads(logical, mem_free):
    """Все логические ядра, кроме одного для интерфейса (если ядер не меньше четырёх), но не больше, чем
    позволяет половина свободной памяти при наибольшей сетке."""
    t = logical - 1 if logical >= 4 else logical
    by_mem = int(0.5 * mem_free * 2**30 // (BYTES_PER_CELL * MAX_CELLS))
    if by_mem < t:
        return max(1, by_mem), f"ограничено свободной памятью ({mem_free:.1f} ГБ)"
    return t, "одно ядро оставлено интерфейсу" if t < logical else "все ядра"


def analyze(threads=None):
    """Анализ и короткий замер; threads задаёт число потоков вручную (None — автоматически)."""
    logical = os.cpu_count() or 1
    mem = free_memory_gb()
    auto, reason = choose_threads(logical, mem)
    t = auto if threads is None else max(1, min(int(threads), logical))
    if threads is not None:
        reason = "задано вручную"
    if _lercore is not None:
        _lercore.set_threads(t)
    # Замер на фрагменте 0,6 мкм; время пропорционально числу ячеек, для 1,2 мкм — вдвое больше.
    p = replace(Process(), L=600.0)
    simulate(replace(p, L=100.0), 1, keep_map=False)
    t0 = time.perf_counter()
    simulate(p, 1, keep_map=False)
    t1 = time.perf_counter() - t0
    tn = t1
    if _lercore is not None and t > 1:
        t0 = time.perf_counter()
        simulate(p, t + 1, keep_map=False)
        tn = (time.perf_counter() - t0 - t1) / t
    t1, tn = 2 * t1, 2 * tn
    if _lercore is not None:
        _lercore.release()
    return Machine(logical, physical_cores(logical), mem, t, t1, max(tn, 1e-4), reason)
