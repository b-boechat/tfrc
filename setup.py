from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "lsm",
        ["lsm.pyx"]
    ),
    Extension(
        "lsm_baseline",
        ["lsm_baseline.pyx"]
    ),

    Extension(
        "lsm_interpol_v1",
        ["lsm_interpol_v1.pyx"]
    ),
    Extension(
        "lsm_baseline_interpol",
        ["lsm_baseline_interpol.pyx"]
    ),
    Extension(
        "lsm_hybrid",
        ["lsm_hybrid.pyx"]
    ),

    Extension(
        "lukin_todd",
        ["lukin_todd.pyx"]
    ),
    Extension(
        "lukin_todd_v1",
        ["lukin_todd_v1.pyx"]
    ),
    Extension(
        "lukin_todd_baseline",
        ["lukin_todd_baseline.pyx"]
    ),

    Extension(
        "fls",
        ["fls.pyx"]
    ),
    Extension(
        "fls_hybrid",
        ["fls_hybrid.pyx"]
    ),
    Extension(
        "fls_hybrid_bin",
        ["fls_hybrid_bin.pyx"]
    ),
    Extension(
        "swgm",
        ["swgm.pyx"]
    )
]

setup(
    name='cython_implementations',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}, annotate=True),
    zip_safe=False
)