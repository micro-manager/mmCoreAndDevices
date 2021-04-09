

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MCCDAQ.cpp'
]


shared_library('MCCDAQ', sources, dependencies : deps, install: true)

