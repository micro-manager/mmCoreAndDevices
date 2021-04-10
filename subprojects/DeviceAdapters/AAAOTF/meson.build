

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AAAOTF.cpp'
]


shared_library('AAAOTF', sources, include_directories: includes, dependencies : deps, install: true)

