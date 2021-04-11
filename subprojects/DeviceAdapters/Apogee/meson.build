

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Apogee.cpp'
]


shared_library('Apogee', sources, include_directories: includes, dependencies : deps, install: true)

