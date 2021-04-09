

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SutterStage.cpp'
]


shared_library('SutterStage', sources, dependencies : deps, install: true)

