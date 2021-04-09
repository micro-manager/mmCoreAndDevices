

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DemoCamera.cpp'
]


shared_library('DemoCamera', sources, dependencies : deps, install: true)

