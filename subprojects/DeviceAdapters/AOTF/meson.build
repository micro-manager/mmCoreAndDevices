

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AOTF.cpp'
]


shared_library('AOTF', sources, dependencies : deps, install: true)

