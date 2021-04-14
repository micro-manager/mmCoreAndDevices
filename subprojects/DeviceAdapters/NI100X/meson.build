

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'NI100X.cpp'
]


library('NI100X', sources, include_directories: includes, dependencies : deps, install: true)

