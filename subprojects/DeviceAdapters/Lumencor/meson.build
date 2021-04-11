

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Lumencor.cpp'
]


shared_library('Lumencor', sources, include_directories: includes, dependencies : deps, install: true)

