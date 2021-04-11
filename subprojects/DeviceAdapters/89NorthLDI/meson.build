

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LDI.cpp'
]


shared_library('89NorthLDI', sources, include_directories: includes, dependencies : deps, install: true)

