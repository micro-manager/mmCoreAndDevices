

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LDI.cpp'
]


shared_library('89NorthLDI', sources, dependencies : deps, install: true)

