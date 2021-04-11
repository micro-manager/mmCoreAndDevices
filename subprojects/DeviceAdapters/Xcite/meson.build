

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Xcite.cpp'
]


shared_library('Xcite', sources, include_directories: includes, dependencies : deps, install: true)

