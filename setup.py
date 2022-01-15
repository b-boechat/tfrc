from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "lsm_par",
        ["lsm_par.pyx"],
        extra_compile_args=['/openmp']
        #extra_link_args=['-fopenmp'],
    ),
    Extension(
        "lsm",
        ["lsm.pyx"],
    )
]

setup(
    name='lsm',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}, annotate=True),
    zip_safe=False
)