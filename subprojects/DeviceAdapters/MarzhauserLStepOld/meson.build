

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LStepOld.cpp'
]


shared_library('MarzhauserLStepOld', sources, dependencies : deps, install: true)

