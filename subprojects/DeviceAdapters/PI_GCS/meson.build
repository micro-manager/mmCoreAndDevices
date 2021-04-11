

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PI_GCS.cpp'
]


shared_library('PI_GCS', sources, include_directories: includes, dependencies : deps, install: true)

