

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'dc1394.cpp'
]


shared_library('dc1394', sources, include_directories: includes, dependencies : deps, install: true)

