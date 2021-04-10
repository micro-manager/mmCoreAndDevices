

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LambdaParallel.cpp'
]


shared_library('SutterLambdaParallelArduino', sources, include_directories: includes, dependencies : deps, install: true)

