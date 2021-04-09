

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'nPC400.cpp',
	'nPC400Channel.cpp',
	'nPC400Ctrl.cpp'
]


shared_library('nPoint', sources, dependencies : deps, install: true)

