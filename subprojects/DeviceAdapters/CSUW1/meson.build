

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CSUW1.cpp',
	'CSUW1Hub.cpp'
]


shared_library('CSUW1', sources, dependencies : deps, install: true)

