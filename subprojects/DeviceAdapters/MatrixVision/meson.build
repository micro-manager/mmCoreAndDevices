

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'mvIMPACT_Acquire_Device.cpp'
]


shared_library('MatrixVision', sources, dependencies : deps, install: true)

