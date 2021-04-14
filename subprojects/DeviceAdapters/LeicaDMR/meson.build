

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LeicaDMR.cpp',
	'LeicaDMRHub.cpp'
]


library('LeicaDMR', sources, include_directories: includes, dependencies : deps, install: true)

