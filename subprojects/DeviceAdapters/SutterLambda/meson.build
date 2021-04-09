

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SutterLambda.cpp'
]


shared_library('SutterLambda', sources, dependencies : deps, install: true)

