from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

ext = Extension("cython_merge", ["cython_merge.pyx"], include_dirs=[np.get_include()])

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
