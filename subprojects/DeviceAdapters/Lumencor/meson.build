

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Lumencor.cpp'
]


shared_library('Lumencor', sources, dependencies : deps, install: true)

