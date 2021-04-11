

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'niji.cpp'
]


shared_library('BlueboxOptics_niji', sources, include_directories: includes, dependencies : deps, install: true)

