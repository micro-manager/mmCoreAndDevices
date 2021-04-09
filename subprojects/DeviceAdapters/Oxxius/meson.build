

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Oxxius.cpp'
]


shared_library('Oxxius', sources, dependencies : deps, install: true)

