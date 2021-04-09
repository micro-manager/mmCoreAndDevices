

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SmarActHCU-3D.cpp'
]


shared_library('SmarActHCU-3D', sources, dependencies : deps, install: true)

