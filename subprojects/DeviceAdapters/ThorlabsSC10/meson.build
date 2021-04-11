

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SC10.cpp'
]


shared_library('ThorlabsSC10', sources, include_directories: includes, dependencies : deps, install: true)

