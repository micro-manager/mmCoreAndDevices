

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CSU22.cpp',
	'CSU22Hub.cpp',
	'CSUX.cpp',
	'CSUXHub.cpp'
]


shared_library('Yokogawa', sources, include_directories: includes, dependencies : deps, install: true)

