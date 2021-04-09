

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'microz.cpp'
]


shared_library('SouthPort', sources, dependencies : deps, install: true)

