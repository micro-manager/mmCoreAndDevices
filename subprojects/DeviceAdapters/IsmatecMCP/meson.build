

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


shared_library('IsmatecMCP', sources, dependencies : deps, install: true)

