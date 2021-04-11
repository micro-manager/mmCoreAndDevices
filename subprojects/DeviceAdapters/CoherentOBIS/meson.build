

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CoherentOBIS.cpp'
]


shared_library('CoherentOBIS', sources, include_directories: includes, dependencies : deps, install: true)

