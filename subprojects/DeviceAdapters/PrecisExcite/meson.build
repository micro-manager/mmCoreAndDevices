

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PrecisExcite.cpp'
]


shared_library('PrecisExcite', sources, include_directories: includes, dependencies : deps, install: true)

