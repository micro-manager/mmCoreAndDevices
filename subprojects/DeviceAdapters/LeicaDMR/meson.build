

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LeicaDMR.cpp',
	'LeicaDMRHub.cpp'
]


shared_library('LeicaDMR', sources, dependencies : deps, install: true)

