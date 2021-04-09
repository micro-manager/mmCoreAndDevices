# -*- coding: utf-8 -*-
"""
This script is called from meson.build in order to dynamically determine
which subdirectories are present.

@author: nick
"""
from glob import glob
import os

wDir = os.path.split(__file__)[0]
files = glob(os.path.join(wDir, '*'))
files = [f for f in files if os.path.isdir(f)]
files = [f[len(wDir)+1:] for f in files]
for f in files:
    print(f)