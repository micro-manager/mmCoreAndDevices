

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AOTF.cpp'
]


shared_library('AOTF', sources, include_directories: includes, dependencies : deps, install: true)

