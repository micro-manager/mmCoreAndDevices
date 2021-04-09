

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Pecon.cpp'
]


shared_library('Pecon', sources, dependencies : deps, install: true)

