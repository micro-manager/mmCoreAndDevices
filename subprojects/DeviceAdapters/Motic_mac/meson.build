

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MUCamSource.cpp'
]


shared_library('Motic_mac', sources, include_directories: includes, dependencies : deps, install: true)

