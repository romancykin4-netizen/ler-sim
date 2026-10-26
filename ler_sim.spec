# python -m PyInstaller --noconfirm --clean ler_sim.spec
a = Analysis(
    ["ler_gui.py"],
    pathex=["."],
    hiddenimports=["lersim._lercore"],
    datas=[("docs/руководство.html", "docs")],
    excludes=["tkinter", "streamlit", "scipy", "IPython", "pytest", "PyQt5", "PyQt6", "pybind11",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick", "PySide6.QtPdf",
              "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LER_sim", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="LER_sim", upx=False)
