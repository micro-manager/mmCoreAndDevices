

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'BaslerPylonCamera.cpp'
]


shared_library('Basler', sources, dependencies : deps, install: true)

