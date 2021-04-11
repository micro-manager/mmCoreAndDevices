

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PvDebayer.cpp',
	'QICamera.cpp'
]


shared_library('QCam', sources, include_directories: includes, dependencies : deps, install: true)

