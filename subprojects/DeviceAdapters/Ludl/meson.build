

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Ludl.cpp'
]


shared_library('Ludl', sources, include_directories: includes, dependencies : deps, install: true)

