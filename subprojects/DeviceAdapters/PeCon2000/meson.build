

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PeCon2000.cpp'
]


shared_library('PeCon2000', sources, dependencies : deps, install: true)

