

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'devicelist.cpp',
	'Piezosystem_dDrive.cpp'
]


shared_library('Piezosystem_dDrive', sources, include_directories: includes, dependencies : deps, install: true)

