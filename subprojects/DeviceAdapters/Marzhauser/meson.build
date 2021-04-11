

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Marzhauser.cpp'
]


shared_library('Marzhauser', sources, include_directories: includes, dependencies : deps, install: true)

