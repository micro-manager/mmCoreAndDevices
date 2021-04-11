

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DemoCamera.cpp'
]


shared_library('DemoCamera', sources, include_directories: includes, dependencies : deps, install: true)

