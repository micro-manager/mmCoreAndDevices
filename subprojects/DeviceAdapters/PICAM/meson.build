

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PICAMAdapter.cpp',
	'PICAMParam.cpp',
	'PICAMUniversal.cpp'
]


library('PICAM', sources, include_directories: includes, dependencies : deps, install: true)

