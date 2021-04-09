

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'VariLC.cpp'
]


shared_library('VariLC', sources, dependencies : deps, install: true)

