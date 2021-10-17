# JetMonteCarlo - Python package for high-energy particle physics.
# Copyright (C) 2020-2021 Samuel Alipour-fard

import re
import sys

from setuptools import setup

with open('jetmontecarlo/__init__.py', 'r') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(version=__version__)
