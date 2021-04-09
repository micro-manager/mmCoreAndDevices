

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'wptr.cpp'
]


shared_library('ASIWPTR', sources, dependencies : deps, install: true)

