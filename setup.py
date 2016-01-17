#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import dirname, realpath
from setuptools import setup, Command
import subprocess
import sys


author = u"Paul Müller"
authors = [author]
name = 'miefield'
description = '2d scattering code'
year = "2015"

long_description = """
This is an attempted pythonification of the Mie code from Guangran Kevin Zhu at 
http://www.mathworks.de/matlabcentral/fileexchange/30162-cylinder-scattering and
http://de.mathworks.com/matlabcentral/fileexchange/31119-sphere-scattering
"""

import miefield
version=miefield.__version__


class PyTest(Command):
    """ Perform pytests
    """
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call([sys.executable, 'tests/runtests.py'])
        raise SystemExit(errno)


if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email='paul.mueller at biotec.tu-dresden.de',
        url='http://RI-imaging.github.io/miefield/',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=long_description,
        install_requires=["NumPy>=1.7.0", "SciPy>=0.10.0"],
        platforms=['ALL'],
        cmdclass = {'test': PyTest,
                    },
        )
