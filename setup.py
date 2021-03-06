from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "lsm",
        ["lsm.pyx"]
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
        "swgm",
        ["swgm.pyx"]
    )
]

setup(
    name='cython_implementations',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}, annotate=True),
    zip_safe=False
)