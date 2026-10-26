"""Сборка C++-ядра в lersim/_lercore.*: python lercore/build.py"""
import sys
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

here = Path(__file__).resolve().parent
msvc = sys.platform == "win32"
ext = Pybind11Extension("lersim._lercore", [str(here / "lercore.cpp")], cxx_std=17,
                        extra_compile_args=["/O2", "/openmp", "/utf-8"] if msvc else ["-O3", "-fopenmp"],
                        extra_link_args=[] if msvc else ["-fopenmp"])
cache = here.parent / ".cache"
sys.argv[1:] = ["build_ext", "--inplace", "--build-temp", str(cache / "cbuild"), "--build-lib", str(cache / "clib")]
setup(name="lercore", ext_modules=[ext], cmdclass={"build_ext": build_ext},
      script_args=sys.argv[1:], package_dir={"": str(here.parent)})
