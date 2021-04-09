

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Newport.cpp'
]


shared_library('NewportSMC', sources, dependencies : deps, install: true)

