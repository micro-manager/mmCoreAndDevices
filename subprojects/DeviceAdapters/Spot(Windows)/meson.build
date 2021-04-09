

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CodeUtility.cpp',
	'SpotCamera.cpp',
	'SpotDevice.cpp'
]


shared_library('Spot(Windows)', sources, dependencies : deps, install: true)

