"""Упаковывает dist\\LER_sim в dist\\LER_sim_win64.zip; повторяет чтение файлов, занятых антивирусом."""
import sys, time, zipfile
from pathlib import Path

src = Path(__file__).resolve().parent / "dist" / "LER_sim"
out = src.parent / "LER_sim_win64.zip"
if not (src / "LER_sim.exe").exists():
    sys.exit(f"Нет {src / 'LER_sim.exe'}: сначала соберите программу")
tmp = out.with_suffix(".tmp")
files = [f for f in sorted(src.rglob("*")) if f.is_file()]
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        for attempt in range(20):
            try:
                z.write(f, Path("LER_sim") / f.relative_to(src))
                break
            except PermissionError:
                time.sleep(0.5)
        else:
            tmp.unlink(missing_ok=True)
            sys.exit(f"Файл занят другим процессом: {f}")
with zipfile.ZipFile(tmp) as z:
    bad = z.testzip()
    if bad:
        sys.exit(f"Архив повреждён на файле {bad}")
tmp.replace(out)
print(f"Архив: {out} ({out.stat().st_size / 2**20:.0f} МБ, файлов: {len(files)})")
