

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Stradus.cpp',
	'VersaLase.cpp'
]


shared_library('Vortran', sources, include_directories: includes, dependencies : deps, install: true)

