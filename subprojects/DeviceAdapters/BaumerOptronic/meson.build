

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'BaumerOptronic.cpp'
]


shared_library('BaumerOptronic', sources, include_directories: includes, dependencies : deps, install: true)

