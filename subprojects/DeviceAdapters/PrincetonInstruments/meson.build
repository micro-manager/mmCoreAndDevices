

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PVCAM.cpp',
	'PVCAMUniversal.cpp',
	'PVCAMUtils.cpp'
]


shared_library('PrincetonInstruments', sources, include_directories: includes, dependencies : deps, install: true)

