

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CSUW1.cpp',
	'CSUW1Hub.cpp'
]


shared_library('CSUW1', sources, include_directories: includes, dependencies : deps, install: true)

