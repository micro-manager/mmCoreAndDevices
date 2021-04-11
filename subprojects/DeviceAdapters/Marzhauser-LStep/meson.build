

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LStep.cpp'
]


shared_library('Marzhauser-LStep', sources, include_directories: includes, dependencies : deps, install: true)

