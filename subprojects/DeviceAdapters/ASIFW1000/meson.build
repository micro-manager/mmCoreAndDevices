

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ASIFW1000.cpp',
	'ASIFW1000Hub.cpp'
]


library('ASIFW1000', sources, include_directories: includes, dependencies : deps, install: true)

