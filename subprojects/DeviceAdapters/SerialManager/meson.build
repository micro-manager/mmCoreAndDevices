

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SerialManager.cpp'
]


shared_library('SerialManager', sources, dependencies : deps, install: true)

