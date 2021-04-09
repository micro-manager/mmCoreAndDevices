

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'niji.cpp'
]


shared_library('BlueboxOptics_niji', sources, dependencies : deps, install: true)

