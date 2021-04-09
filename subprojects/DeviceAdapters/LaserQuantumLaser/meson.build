

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LaserQuantumLaser.cpp'
]


shared_library('LaserQuantumLaser', sources, dependencies : deps, install: true)

