

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'K8055.cpp',
	'K8055Interface.cpp'
]


shared_library('K8055', sources, dependencies : deps, install: true)

