

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'E600.cpp'
]


shared_library('YodnE600', sources, dependencies : deps, install: true)

