

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'nPC400.cpp',
	'nPC400Channel.cpp',
	'nPC400Ctrl.cpp'
]


shared_library('nPoint', sources, include_directories: includes, dependencies : deps, install: true)

