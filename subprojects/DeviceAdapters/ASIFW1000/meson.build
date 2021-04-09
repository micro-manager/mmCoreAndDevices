

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ASIFW1000.cpp',
	'ASIFW1000Hub.cpp'
]


shared_library('ASIFW1000', sources, dependencies : deps, install: true)

