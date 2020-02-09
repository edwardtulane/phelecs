
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

correlations = Extension(  'phelecs.correlations',
                          ['phelecs/correlations.pyx'],
                        )


setup(
  name = 'phelecs',
  version='0.1',
  ext_modules = cythonize(correlations),
                 
# ext_modules = [abel],
  packages = ['phelecs'],
# package_dir = {'': 'lib'},
  include_dirs=[numpy.get_include()],
  
)
