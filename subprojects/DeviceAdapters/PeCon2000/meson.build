

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PeCon2000.cpp'
]


shared_library('PeCon2000', sources, include_directories: includes, dependencies : deps, install: true)

