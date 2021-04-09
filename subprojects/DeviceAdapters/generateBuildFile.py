# -*- coding: utf-8 -*-
"""
This script can be used to generate a default `meson.build` file for
a device adapter folder.

The optional `--search_pattern` argument can be used to only generate 
files for subfolders that match the search pattern. For example:
    `python generateBuildFile.py --search_pattern=Vari*`
will only generate build files for folders that start with `Vari`

If `--search_pattern` is not specified then files will be generated for
all subfolders.

The generated build file will include MMDevice and Boost as dependencies,
will add the root folder as an include path, and will all all .cpp files
as sources.

@author: Nick Anthony
"""
import os
from glob import glob
import sys
import argparse

parser = argparse.ArgumentParser(description="Determine the glob pattern to search for subdirectories.")
parser.add_argument('--search_pattern', type=str, default="*", required=False)

args = parser.parse_args()
searchPattern = args.search_pattern
print(f"Search pattern is: {searchPattern}")

template = """

includes = include_directories('.') # include the main directory

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	{sources}
]


shared_library('{subdir}', sources, include_directories: includes, dependencies : deps, install: true)

"""

wDir = os.path.split(__file__)[0]



for f in glob(os.path.join(wDir, searchPattern)):
    if os.path.isdir(f):
        subdir = f[len(wDir)+1:]
        print(f"Generating build file for: {subdir}")
        sources = glob(os.path.join(f, '**.cpp'), recursive=True) 
        sources = [s[len(f)+1:] for s in sources]
        sources = ["'" + s + "'" for s in sources]
        s = template.format(subdir=subdir, sources=',\n\t'.join(sources))
        with open(os.path.join(f, 'meson.build'), 'w') as file:
            file.write(s)
        
