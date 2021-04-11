

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Conex_Axis.cpp'
]


shared_library('NewportCONEX', sources, include_directories: includes, dependencies : deps, install: true)

