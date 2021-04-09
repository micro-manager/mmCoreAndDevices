

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Nikon.cpp'
]


shared_library('Nikon', sources, dependencies : deps, install: true)

