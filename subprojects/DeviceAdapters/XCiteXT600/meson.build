

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XT600.cpp',
	'XT600Ctrl.cpp',
	'XT600Dev.cpp'
]


shared_library('XCiteXT600', sources, dependencies : deps, install: true)

