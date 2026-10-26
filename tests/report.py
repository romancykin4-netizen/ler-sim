"""Отчёт об испытаниях программной реализации модели: HTML с подробными результатами каждой проверки
и CSV для Excel. Оформление — по образцу отчёта о НИР (ГОСТ 7.32): нумерованные разделы, подписи таблиц
«Таблица N — …», десятичная запятая, степени вида 1·10⁻¹².

Данные берутся из самих тестов: структурированная строка документации (заголовок, «Метод:»,
«Критерий:», «Зачем:») и измерения, записанные фикстурой rec. Ничего не дописывается вручную.
"""
import csv, html, inspect, os, platform, re, sys, textwrap
from datetime import datetime
from pathlib import Path

KIND = {"white": "белый ящик", "black": "чёрный ящик"}
KIND_NAME = {"белый ящик": "структурное (метод белого ящика)", "чёрный ящик": "функциональное (метод чёрного ящика)"}
KIND_TEXT = {
    "белый ящик": "Проверка с учётом внутреннего устройства: отдельная функция сравнивается с точным (аналитическим) решением.",
    "чёрный ящик": "Проверка без учёта внутреннего устройства: по входным данным и результату расчёта проверяются свойства результата.",
}
STATUS = {"passed": "пройдена", "failed": "не пройдена", "skipped": "не выполнялась"}
SECTIONS = ("Метод", "Критерий", "Зачем")
SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def sci(text):
    """Запись чисел по нормам научных текстов: 1e-12 → 1·10⁻¹², десятичная запятая."""
    text = str(text)

    def rep(m):
        mant, exp = m.group(1), int(m.group(2))
        mant = mant.rstrip("0").rstrip(".") if "." in mant else mant
        return ("" if mant == "1" else mant + "·") + "10" + str(exp).translate(SUP)
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]([-+]?\d+)(?![\w.])", rep, text)
    return re.sub(r"(?<=\d)\.(?=\d)", ",", text)


def fmt(v):
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "—"
        a = abs(v)
        return sci(f"{v:.3e}" if (a and (a < 1e-3 or a >= 1e6)) else f"{v:.4g}").replace("-", "−")
    return sci(v)


def parse_doc(doc):
    """Заголовок — первый абзац, разделы — строки «Метод:», «Критерий:», «Зачем:», остальное — описание."""
    out = dict(title="", text="", **{k: "" for k in SECTIONS})
    if not doc:
        return out
    doc = textwrap.dedent(doc).strip()
    paras = re.split(r"\n\s*\n", doc)
    out["title"] = " ".join(paras[0].split())
    cur, free = None, []
    for line in "\n".join(paras[1:]).splitlines():
        m = re.match(r"\s*(Метод|Критерий|Зачем):\s*(.*)", line)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2).strip()
        elif cur and line.strip():
            out[cur] += " " + line.strip()
        elif line.strip():
            free.append(line.strip())
    out["text"] = " ".join(free)
    return out


class Collector:
    def __init__(self):
        self.reports = {}

    def add(self, rep):
        if rep.when == "call" or rep.outcome != "passed":
            prev = self.reports.get(rep.nodeid)
            if prev is None or prev.outcome == "passed":
                self.reports[rep.nodeid] = rep

    def rows(self, session):
        items = {i.nodeid: i for i in session.items}
        rows = []
        for nodeid, rep in self.reports.items():
            it = items.get(nodeid)
            fn = getattr(it, "function", None)
            mod = getattr(it, "module", None)
            d = parse_doc(fn.__doc__ if fn else "")
            path = nodeid.replace("\\", "/")
            kind = next((v for k, v in KIND.items() if f"tests/{k}/" in path), "")
            recs = [v for k, v in rep.user_properties if k == "rec"]
            params = getattr(getattr(it, "callspec", None), "params", {}) or {}
            err = ""
            if rep.outcome == "failed":
                err = str(rep.longrepr)
            elif rep.outcome == "skipped":
                lr = rep.longrepr
                err = lr[2] if isinstance(lr, tuple) and len(lr) == 3 else str(lr)
                err = err.replace("Skipped: ", "")
            try:
                src = textwrap.dedent(inspect.getsource(fn)) if fn else ""
            except (OSError, TypeError):
                src = ""
            d["title"] = d["title"].rstrip(".")
            rows.append(dict(
                nodeid=nodeid, name=nodeid.split("::")[-1], file=path.split("::")[0], kind=kind,
                module=" ".join(re.split(r"\n\s*\n", (mod.__doc__ or "").strip())[0].split()) if mod else "",
                params=", ".join(f"{k} = {fmt(v)}" for k, v in params.items()), status=rep.outcome,
                duration=rep.duration, recs=recs, error=err, source=src, **d))
        order = {n: i for i, n in enumerate(items)}
        rows.sort(key=lambda r: order.get(r["nodeid"], 1e9))
        for i, r in enumerate(rows, 1):
            r["no"] = i
        return rows


def environment():
    env = [("Дата и время", f"{datetime.now():%d.%m.%Y %H:%M:%S}"),
           ("Компьютер", platform.node()), ("Операционная система", f"{platform.system()} {platform.release()} ({platform.machine()})"),
           ("Число логических процессоров", str(os.cpu_count())), ("Интерпретатор Python", sys.version.split()[0])]
    for mod in ("numpy", "matplotlib", "PySide6", "pytest"):
        try:
            env.append((mod, __import__(mod).__version__))
        except Exception:
            env.append((mod, "не установлен"))
    root = Path(__file__).resolve().parents[1]
    m = re.search(r'VERSION\s*=\s*"([^"]+)"', (root / "ler_gui.py").read_text(encoding="utf-8")) if (root / "ler_gui.py").exists() else None
    env.append(("Версия программы", m.group(1) if m else "—"))
    try:
        from lersim.process import _lercore
        env.append(("Вычислительное ядро C++", f"собрано, потоков OpenMP: {_lercore.threads()}" if _lercore else "не собрано (проверки ядра не выполнялись)"))
    except Exception as e:
        env.append(("Вычислительное ядро C++", f"ошибка загрузки: {e}"))
    g = root / "tests" / "data" / "golden.json"
    if g.exists():
        env.append(("Эталонные значения (golden.json)", f"от {datetime.fromtimestamp(g.stat().st_mtime):%d.%m.%Y %H:%M}"))
    return env


def plural(n, one, few, many):
    n10, n100 = n % 10, n % 100
    return one if n10 == 1 and n100 != 11 else few if 2 <= n10 <= 4 and not 12 <= n100 <= 14 else many


def conclusion(r):
    bad = [x for x in r["recs"] if x["ok"] is False]
    n = len(r["recs"])
    if r["status"] == "passed":
        if not n:
            return "Требования критерия приёмки выполнены."
        k = sum(x["ok"] is not None for x in r["recs"])
        s = f"Требования критерия приёмки выполнены. Зарегистрировано {n} {plural(n, 'значение', 'значения', 'значений')}"
        return s + (f", из них {k} с нормируемым требованием — все соответствуют." if k else ".")
    if r["status"] == "skipped":
        return f"Проверка не выполнялась: {sci(r['error'])}."
    msg = "Требования критерия приёмки не выполнены."
    if bad:
        msg += " Не соответствуют требованиям: " + "; ".join(
            f"{x['name']} = {fmt(x['value'])}{(' ' + x['unit']) if x['unit'] else ''} (требование: {sci(x['expect'])})" for x in bad) + "."
    else:
        last = [l for l in r["error"].splitlines() if l.startswith("E ")]
        if last:
            msg += " " + last[0][1:].strip()
    return msg


def verdict(ok):
    return "" if ok is None else ("соответствует" if ok else "не соответствует")


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["№ п/п", "Вид испытаний", "Модуль", "Идентификатор проверки", "Параметры", "Наименование проверки",
                    "Методика", "Критерий приёмки", "Цель проверки", "Результат", "Длительность, с",
                    "Зарегистрировано значений", "Не соответствует требованиям", "Контролируемые величины", "Заключение"])
        for r in rows:
            meas = " | ".join(f"{x['name']} = {fmt(x['value'])}{(' ' + x['unit']) if x['unit'] else ''}"
                              f"{(' (требование: ' + sci(x['expect']) + ')') if x['expect'] else ''}" for x in r["recs"])
            w.writerow([r["no"], KIND_NAME.get(r["kind"], r["kind"]), r["file"], r["name"], sci(r["params"]), sci(r["title"]),
                        sci(r["Метод"]), sci(r["Критерий"]), sci(r["Зачем"]), STATUS[r["status"]],
                        f"{r['duration']:.3f}".replace(".", ","), len(r["recs"]),
                        sum(x["ok"] is False for x in r["recs"]), meas, conclusion(r)])
    mpath = path.with_name(path.stem + "_измерения.csv")
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["№ проверки", "Вид испытаний", "Идентификатор проверки", "Параметры", "Контролируемая величина",
                    "Измеренное значение", "Единица измерения", "Требование", "Соответствие", "Результат проверки"])
        for r in rows:
            for x in r["recs"]:
                w.writerow([r["no"], KIND_NAME.get(r["kind"], r["kind"]), r["name"], sci(r["params"]), sci(x["name"]),
                            fmt(x["value"]), x["unit"], sci(x["expect"]), verdict(x["ok"]), STATUS[r["status"]]])
    return mpath


CSS = """
:root{--fg:#111;--mut:#555;--line:#999;--soft:#f2f2f2;--ok:#1d5e33;--bad:#9b1c12;--sk:#6b5200;--acc:#1b3a6b}
*{box-sizing:border-box}html{background:#e9e9e9}
body{margin:0;color:var(--fg);font:15px/1.5 "Times New Roman",Times,"Liberation Serif",serif}
main{max-width:1060px;margin:24px auto;background:#fff;padding:48px 64px 64px;box-shadow:0 0 0 1px #d6d6d6}
.title{text-align:center;margin-bottom:28px}.title .org{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.title h1{font-size:20px;text-transform:uppercase;margin:18px 0 6px;letter-spacing:.02em}.title .sub{font-size:16px}
h2{font-size:16px;margin:30px 0 10px;text-transform:uppercase}h3{font-size:15px;margin:22px 0 6px}
p{margin:6px 0;text-align:justify}.mut{color:var(--mut)}
.cap{margin:14px 0 4px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:6px}
th,td{border:1px solid var(--line);padding:3px 6px;vertical-align:top;text-align:left}
th{background:var(--soft);font-weight:bold;text-align:center}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}td.c{text-align:center}
.st{white-space:nowrap}.st.passed{color:var(--ok)}.st.failed{color:var(--bad);font-weight:bold}.st.skipped{color:var(--sk)}
td.ok{color:var(--ok)}td.bad{color:var(--bad);font-weight:bold}
.card{margin:18px 0 26px;break-inside:avoid}.card h3 .id{font:12px Consolas,monospace;color:var(--mut);font-weight:normal;margin-left:6px}
table.kv th{width:210px;text-align:left;font-weight:normal;background:var(--soft)}
pre{background:#f7f7f7;border:1px solid #ddd;padding:8px;overflow:auto;font:12px/1.4 Consolas,monospace;white-space:pre-wrap}
details summary{cursor:pointer;color:var(--acc);font-size:14px}a{color:var(--acc)}
.filters{margin:6px 0;font-size:14px}.filters label{margin-right:14px}
@media (max-width:760px){main{padding:24px 16px;margin:0}table.kv th{width:auto}}
@media print{html{background:#fff}main{box-shadow:none;margin:0;padding:0;max-width:none}.filters,details{display:none}
 .card{page-break-inside:avoid}h2{page-break-after:avoid}a{color:inherit;text-decoration:none}}
"""

JS = """
const boxes=document.querySelectorAll('.filters input');
function apply(){const on=[...boxes].filter(b=>b.checked).map(b=>b.value);
document.querySelectorAll('[data-status]').forEach(e=>e.style.display=on.includes(e.dataset.status)?'':'none');}
boxes.forEach(b=>b.addEventListener('change',apply));
"""


def write_html(rows, path, duration, csv_path, mcsv_path):
    e = lambda t: html.escape(sci(t))
    cnt = {s: sum(r["status"] == s for r in rows) for s in STATUS}
    total, nrec = len(rows), sum(len(r["recs"]) for r in rows)
    nbad = sum(x["ok"] is False for r in rows for x in r["recs"])
    now = datetime.now()
    tab = [0]

    def cap(text):
        tab[0] += 1
        return f"<p class='cap'>Таблица {tab[0]} — {html.escape(text)}</p>"

    out = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Отчёт об испытаниях модели LER</title><style>{CSS}</style></head><body><main>",
           "<div class='title'><div class='org'>Программа «Предиктивное моделирование LER при фотолитографии»</div>"
           "<h1>Отчёт об испытаниях</h1><div class='sub'>программной реализации модели шероховатости края и ширины линии (LER/LWR)</div>"
           f"<div class='mut'>{now:%d.%m.%Y}, {now:%H:%M}</div></div>"]

    out.append("<h2>1 Общие сведения</h2>")
    out.append("<p>Объект испытаний — программная реализация стохастической модели формирования края линии в химически "
               "усиленном резисте (DUV, 193/248 нм): вычислительное ядро C++, эталонная реализация на NumPy, "
               "расчёт окна процесса и пользовательский интерфейс.</p>")
    out.append("<p>Цель испытаний — подтвердить, что программа правильно вычисляет заложенную модель и что результаты "
               "подчиняются известным физическим закономерностям.</p>")
    out.append("<p>Применены два вида испытаний: структурное (метод белого ящика) — сравнение отдельных функций с "
               "аналитическими решениями; функциональное (метод чёрного ящика) — проверка свойств результата расчёта "
               "по входным данным, без учёта внутреннего устройства.</p>")
    out.append(cap("Условия проведения испытаний"))
    out.append("<table>" + "".join(f"<tr><th style='text-align:left;width:40%'>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                                   for k, v in environment() + [("Продолжительность испытаний", sci(f"{duration:.1f} с"))]) + "</table>")

    out.append("<h2>2 Сводные результаты</h2>")
    out.append(cap("Сводные результаты испытаний"))
    out.append("<table><tr><th>Показатель</th><th>Значение</th></tr>" + "".join(
        f"<tr><td>{k}</td><td class='n'>{v}</td></tr>" for k, v in (
            ("Всего проверок", total), ("Проверки пройдены", cnt["passed"]), ("Проверки не пройдены", cnt["failed"]),
            ("Проверки не выполнялись", cnt["skipped"]), ("Зарегистрировано контролируемых значений", nrec),
            ("Значений, не соответствующих требованиям", nbad))) + "</table>")
    out.append(cap("Результаты по разделам испытаний"))
    out.append("<table><tr><th>Вид испытаний</th><th>Раздел</th><th>Проверок</th><th>Пройдено</th>"
               "<th>Не пройдено</th><th>Не выполнялось</th><th>Длительность, с</th></tr>")
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["file"], r["module"]), []).append(r)
    for (k, f, m), g in groups.items():
        out.append(f"<tr><td>{e(KIND_NAME.get(k, k))}</td><td>{e(m)}<br><span class='mut'>{e(f)}</span></td>"
                   f"<td class='n'>{len(g)}</td>" + "".join(f"<td class='n'>{sum(x['status'] == s for x in g)}</td>" for s in STATUS)
                   + f"<td class='n'>{sci(f'{sum(x['duration'] for x in g):.2f}')}</td></tr>")
    out.append("</table>")
    out.append(f"<p class='mut'>Табличные данные для обработки: <a href='{html.escape(csv_path.name)}'>{html.escape(csv_path.name)}</a> "
               f"(по проверкам), <a href='{html.escape(mcsv_path.name)}'>{html.escape(mcsv_path.name)}</a> (по контролируемым величинам).</p>")

    out.append("<h2>3 Перечень проверок</h2><div class='filters'>Показывать: " + "".join(
        f"<label><input type='checkbox' value='{s}' checked> {t}</label>" for s, t in STATUS.items()) + "</div>")
    out.append(cap("Перечень проверок и их результаты"))
    out.append("<table><tr><th>№</th><th>Наименование проверки</th><th>Вид испытаний</th><th>Результат</th><th>Длит., с</th></tr>")
    for r in rows:
        title = r["title"] + (f" [{r['params']}]" if r["params"] else "")
        out.append(f"<tr data-status='{r['status']}'><td class='n'>4.{r['no']}</td><td><a href='#t{r['no']}'>{e(title)}</a></td>"
                   f"<td>{'структурное' if r['kind'] == 'белый ящик' else 'функциональное'}</td>"
                   f"<td class='c'><span class='st {r['status']}'>{STATUS[r['status']]}</span></td>"
                   f"<td class='n'>{sci(f'{r['duration']:.2f}')}</td></tr>")
    out.append("</table>")

    out.append("<h2>4 Результаты проверок</h2>")
    for r in rows:
        out.append(f"<section class='card' id='t{r['no']}' data-status='{r['status']}'>"
                   f"<h3>4.{r['no']} {e(r['title'] or r['name'])}<span class='id'>{html.escape(r['nodeid'])}</span></h3>")
        kv = [("Вид испытаний", f"{KIND_NAME.get(r['kind'], r['kind'])}. {KIND_TEXT.get(r['kind'], '')}"),
              ("Объект проверки", r["module"]), ("Параметры", r["params"]), ("Описание", r["text"]),
              ("Методика", r["Метод"]), ("Критерий приёмки", r["Критерий"]), ("Цель проверки", r["Зачем"])]
        out.append("<table class='kv'>" + "".join(f"<tr><th>{k}</th><td>{e(v)}</td></tr>" for k, v in kv if v) + "</table>")
        if r["recs"]:
            out.append(cap(f"Контролируемые величины, проверка 4.{r['no']}"))
            out.append("<table><tr><th>Контролируемая величина</th><th>Измеренное значение</th><th>Ед. изм.</th>"
                       "<th>Требование</th><th>Соответствие</th></tr>")
            for x in r["recs"]:
                cls = "" if x["ok"] is None else ("ok" if x["ok"] else "bad")
                out.append(f"<tr><td>{e(x['name'])}</td><td class='n'>{html.escape(fmt(x['value']))}</td><td class='c'>{e(x['unit'])}</td>"
                           f"<td>{e(x['expect'])}</td><td class='c {cls}'>{verdict(x['ok']) or '—'}</td></tr>")
            out.append("</table>")
        out.append("<table class='kv'>"
                   f"<tr><th>Результат</th><td><span class='st {r['status']}'>Проверка {STATUS[r['status']]}</span></td></tr>"
                   f"<tr><th>Заключение</th><td>{html.escape(conclusion(r))}</td></tr>"
                   f"<tr><th>Длительность</th><td>{sci(f'{r['duration']:.3f}')} с</td></tr></table>")
        if r["status"] == "failed":
            opened = "" if any(x["ok"] is False for x in r["recs"]) else " open"
            out.append(f"<details{opened}><summary>Диагностический вывод pytest</summary><pre>{html.escape(r['error'][-4000:])}</pre></details>")
        if r["source"]:
            out.append(f"<details><summary>Программный код проверки</summary><pre>{html.escape(r['source'])}</pre></details>")
        out.append("</section>")

    out.append("<h2>5 Заключение</h2>")
    if cnt["failed"] == 0:
        out.append(f"<p>Выполнено {cnt['passed']} {plural(cnt['passed'], 'проверка', 'проверки', 'проверок')} из {total}; "
                   f"все требования критериев приёмки выполнены, {nrec} {plural(nrec, 'контролируемое значение', 'контролируемых значения', 'контролируемых значений')} "
                   "соответствуют требованиям."
                   + (f" {cnt['skipped']} {plural(cnt['skipped'], 'проверка не выполнялась', 'проверки не выполнялись', 'проверок не выполнялись')} — см. раздел 4." if cnt["skipped"] else "")
                   + " Программная реализация модели признаётся работоспособной.</p>")
    else:
        failed = ", ".join(f"4.{r['no']}" for r in rows if r["status"] == "failed")
        out.append(f"<p>Из {total} проверок не пройдено {cnt['failed']} ({failed}); "
                   f"{nbad} {plural(nbad, 'значение не соответствует', 'значения не соответствуют', 'значений не соответствуют')} требованиям. "
                   "Программная реализация модели требует доработки.</p>")
    out.append("<p>Испытания подтверждают правильность вычисления заложенной модели и соответствие результатов "
               "физическим закономерностям. Соответствие результатов конкретному технологическому процессу ими не "
               "устанавливается: для этого требуется калибровка модели по экспериментальным данным.</p>")
    out.append(f"<script>{JS}</script></main></body></html>")
    path.write_text("\n".join(out), encoding="utf-8")


def write_txt(rows, path):
    lines = [f"Отчёт об испытаниях модели LER/LWR (частичный прогон), {datetime.now():%d.%m.%Y %H:%M}",
             "Итого: " + ", ".join(f"{t} {sum(r['status'] == s for r in rows)}" for s, t in STATUS.items()), ""]
    for r in rows:
        lines.append(f"{STATUS[r['status']]:15s} {r['duration']:6.2f} с  [{KIND_NAME.get(r['kind'], r['kind'])}]  {r['nodeid']}".replace(".", ",", 1))
        if r["title"]:
            lines.append(f"{'':30s}{sci(r['title'])}")
        if r["status"] != "passed":
            lines.append(f"{'':30s}{conclusion(r)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
