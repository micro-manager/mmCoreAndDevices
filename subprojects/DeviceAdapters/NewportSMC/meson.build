

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Newport.cpp'
]


library('NewportSMC', sources, include_directories: includes, dependencies : deps, install: true)

