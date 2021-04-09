# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:25:41 2021

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