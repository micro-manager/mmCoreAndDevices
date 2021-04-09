

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'pE300.cpp'
]


shared_library('CoolLEDpE300', sources, dependencies : deps, install: true)

