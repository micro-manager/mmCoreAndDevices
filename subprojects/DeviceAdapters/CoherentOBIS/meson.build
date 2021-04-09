

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CoherentOBIS.cpp'
]


shared_library('CoherentOBIS', sources, dependencies : deps, install: true)

