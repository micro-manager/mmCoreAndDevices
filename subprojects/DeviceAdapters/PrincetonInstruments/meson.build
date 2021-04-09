

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PVCAM.cpp',
	'PVCAMUniversal.cpp',
	'PVCAMUtils.cpp'
]


shared_library('PrincetonInstruments', sources, dependencies : deps, install: true)

