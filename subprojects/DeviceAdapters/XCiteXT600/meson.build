

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XT600.cpp',
	'XT600Ctrl.cpp',
	'XT600Dev.cpp'
]


shared_library('XCiteXT600', sources, include_directories: includes, dependencies : deps, install: true)

