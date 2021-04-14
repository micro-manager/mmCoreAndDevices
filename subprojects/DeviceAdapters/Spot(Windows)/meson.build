

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CodeUtility.cpp',
	'SpotCamera.cpp',
	'SpotDevice.cpp'
]


library('Spot(Windows)', sources, include_directories: includes, dependencies : deps, install: true)

