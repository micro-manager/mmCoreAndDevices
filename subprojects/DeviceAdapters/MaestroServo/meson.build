

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MaestroServo.cpp'
]


shared_library('MaestroServo', sources, dependencies : deps, install: true)

