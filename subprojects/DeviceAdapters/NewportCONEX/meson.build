

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Conex_Axis.cpp'
]


shared_library('NewportCONEX', sources, dependencies : deps, install: true)

