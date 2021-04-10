

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'pE4000.cpp'
]


shared_library('CoolLEDpE4000', sources, include_directories: includes, dependencies : deps, install: true)

