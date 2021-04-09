

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FocalPoint.cpp'
]


shared_library('FocalPoint', sources, dependencies : deps, install: true)

