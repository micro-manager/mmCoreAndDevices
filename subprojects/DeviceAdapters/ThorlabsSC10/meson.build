

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SC10.cpp'
]


shared_library('ThorlabsSC10', sources, dependencies : deps, install: true)

