

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MoticCamera.cpp'
]


shared_library('Motic', sources, dependencies : deps, install: true)

