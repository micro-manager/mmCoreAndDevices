

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'mvIMPACT_Acquire_Device.cpp'
]


shared_library('MatrixVision', sources, include_directories: includes, dependencies : deps, install: true)

