

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ThorlabsUSBCamera.cpp'
]


shared_library('ThorlabsUSBCamera', sources, dependencies : deps, install: true)

