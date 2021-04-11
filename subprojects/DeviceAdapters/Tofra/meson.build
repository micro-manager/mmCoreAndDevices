

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Tofra.cpp'
]


shared_library('Tofra', sources, include_directories: includes, dependencies : deps, install: true)

