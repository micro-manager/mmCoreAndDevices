

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DeviceEnumerator.cpp',
	'OpenCVgrabber.cpp'
]


library('OpenCVgrabber', sources, include_directories: includes, dependencies : deps, install: true)

