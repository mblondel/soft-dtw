from __future__ import print_function
import os
import sys
from codecs import open

from distutils.core import setup, Extension
from Cython.Distutils import build_ext

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


DISTNAME = 'soft-dtw'
DESCRIPTION = "Python implementation of soft-DTW"
with open('README.rst', 'r', encoding='utf-8') as rm_file:
    LONG_DESCRIPTION = rm_file.read()
MAINTAINER = 'Mathieu Blondel'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/mblondel/soft-dtw/'
LICENSE = 'Simplified BSD'
DOWNLOAD_URL = 'https://github.com/mblondel/soft-dtw/'
VERSION = '0.1.dev0'

extensions = [
    Extension(
        'sdtw.soft_dtw_fast',
        sources=['sdtw/soft_dtw_fast.pyx'],
        include_dirs=[numpy.get_include()],
    ),
]

if __name__ == '__main__':

    setup(
          name=DISTNAME,
          maintainer=MAINTAINER,
          packages=['sdtw'],
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          ext_modules=extensions,
          install_requires=[
              'scipy', 'numpy', 'cython', 'scikit-learn', 'chainer'
          ],
          setup_requires=['numpy', 'cython'],
          version=VERSION,
          include_dirs=[numpy.get_include()],
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          cmdclass={'build_ext': build_ext},
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers', 'License :: OSI Approved',
              'Programming Language :: C', 'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX', 'Operating System :: Unix',
              'Operating System :: MacOS'
             ]
          )
