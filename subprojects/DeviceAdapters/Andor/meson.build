

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Andor.cpp',
	'ReadModeControl.cpp',
	'SpuriousNoiseFilterControl.cpp',
	'SRRFAndorCamera.cpp',
	'SRRFControl.cpp'
]


shared_library('Andor', sources, include_directories: includes, dependencies : deps, install: true)

