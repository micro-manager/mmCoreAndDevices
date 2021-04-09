

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Apogee.cpp'
]


shared_library('Apogee', sources, dependencies : deps, install: true)

