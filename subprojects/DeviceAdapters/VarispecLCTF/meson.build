

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'VarispecLCTF.cpp'
]


shared_library('VarispecLCTF', sources, include_directories: includes, dependencies : deps, install: true)

