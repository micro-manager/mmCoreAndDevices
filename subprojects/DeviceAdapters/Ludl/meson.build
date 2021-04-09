

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Ludl.cpp'
]


shared_library('Ludl', sources, dependencies : deps, install: true)

