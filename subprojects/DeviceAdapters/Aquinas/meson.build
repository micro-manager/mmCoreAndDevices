

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Aquinas.cpp'
]


shared_library('Aquinas', sources, include_directories: includes, dependencies : deps, install: true)

