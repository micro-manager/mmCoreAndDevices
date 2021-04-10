

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'IsmatecMCP.cpp'
]


shared_library('IsmatecMCP', sources, include_directories: includes, dependencies : deps, install: true)

