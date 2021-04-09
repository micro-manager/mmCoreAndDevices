

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Toptica_iBeamSmartCW.cpp'
]


shared_library('Toptica_iBeamSmartCW', sources, dependencies : deps, install: true)

