

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DynDialogs.cpp',
	'SimplePropertyAccess.cpp',
	'TIScamera.cpp'
]


shared_library('TISCam', sources, include_directories: includes, dependencies : deps, install: true)

