

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LStepOld.cpp'
]


shared_library('MarzhauserLStepOld', sources, include_directories: includes, dependencies : deps, install: true)

