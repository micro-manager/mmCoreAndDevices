

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'VTiSIM.cpp'
]


shared_library('VisiTech_iSIM', sources, include_directories: includes, dependencies : deps, install: true)

