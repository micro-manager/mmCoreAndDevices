

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ServoZ.cpp'
]


shared_library('ServoZDrive', sources, dependencies : deps, install: true)

