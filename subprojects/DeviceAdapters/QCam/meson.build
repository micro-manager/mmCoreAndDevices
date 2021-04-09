

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PvDebayer.cpp',
	'QICamera.cpp'
]


shared_library('QCam', sources, dependencies : deps, install: true)

