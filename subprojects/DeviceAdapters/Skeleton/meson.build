

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Skeleton.cpp'
]


shared_library('Skeleton', sources, include_directories: includes, dependencies : deps, install: true)

