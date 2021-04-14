

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'devicelist.cpp',
	'Piezosystem_dDrive.cpp'
]


library('Piezosystem_dDrive', sources, include_directories: includes, dependencies : deps, install: true)

