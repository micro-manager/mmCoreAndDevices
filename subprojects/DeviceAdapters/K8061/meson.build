

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'K8061.cpp',
	'K8061Interface.cpp'
]


library('K8061', sources, include_directories: includes, dependencies : deps, install: true)

