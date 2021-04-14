

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CameraAdapter.cpp',
	'PiperMmAdapter.cpp'
]


library('Piper', sources, include_directories: includes, dependencies : deps, install: true)

