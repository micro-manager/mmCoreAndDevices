

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'BFCamera.cpp',
	'BFTest.cpp',
	'ImgAccumulator.cpp',
	'TwoPhoton.cpp',
	'VirtualShutter.cpp'
]


library('TwoPhoton', sources, include_directories: includes, dependencies : deps, install: true)

