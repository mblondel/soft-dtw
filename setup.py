from __future__ import print_function
import os.path
import sys

from Cython.Build import cythonize
import setuptools
from setuptools.extension import Extension
from setuptools import setup

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


DISTNAME = 'soft-dtw'
DESCRIPTION = "Python implementation of soft-DTW"
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Mathieu Blondel'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/mblondel/soft-dtw/'
LICENSE = 'Simplified BSD'
DOWNLOAD_URL = 'https://github.com/mblondel/soft-dtw/'
VERSION = '0.1.dev0'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('sdtw')

    return config


if __name__ == '__main__':
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    extensions = [
        Extension(
            'sdtw.soft_dtw_fast',
            ['sdtw/soft_dtw_fast.pyx'],
            include_dirs=[numpy.get_include()],
            library_dirs=[],
        ),
    ]

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          ext_modules=extensions,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          install_requires=['scipy', 'numpy', 'cython', 'scikit-learn'],
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
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
