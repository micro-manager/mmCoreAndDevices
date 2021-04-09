

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MicroManager.cpp'
]


shared_library('PCO_Generic', sources, dependencies : deps, install: true)

