

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LStep.cpp'
]


shared_library('Marzhauser-LStep', sources, dependencies : deps, install: true)

