

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LambdaParallel.cpp'
]


library('SutterLambdaParallelArduino', sources, include_directories: includes, dependencies : deps, install: true)

