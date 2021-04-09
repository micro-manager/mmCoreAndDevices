

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PI_GCS.cpp'
]


shared_library('PI_GCS', sources, dependencies : deps, install: true)

