

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LambdaVF5.cpp',
	'Shutter.cpp',
	'SutterHub.cpp',
	'SutterLambda2.cpp'
]


shared_library('SutterLambda2', sources, include_directories: includes, dependencies : deps, install: true)

