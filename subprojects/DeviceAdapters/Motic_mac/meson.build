

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MUCamSource.cpp'
]


shared_library('Motic_mac', sources, dependencies : deps, install: true)

