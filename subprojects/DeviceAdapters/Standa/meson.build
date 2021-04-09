

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Standa.cpp'
]


shared_library('Standa', sources, dependencies : deps, install: true)

