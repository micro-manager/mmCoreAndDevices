

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


library('PrincetonInstruments', sources, include_directories: includes, dependencies : deps, install: true)

