

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DTOpenLayer.cpp'
]


shared_library('DTOpenLayer', sources, dependencies : deps, install: true)

