

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'KDV.cpp'
]


shared_library('kdv', sources, dependencies : deps, install: true)

