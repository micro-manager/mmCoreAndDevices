

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'K8061.cpp',
	'K8061Interface.cpp'
]


shared_library('K8061', sources, dependencies : deps, install: true)

