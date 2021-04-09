

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'DeviceEnumerator.cpp',
	'OpenCVgrabber.cpp'
]


shared_library('OpenCVgrabber', sources, dependencies : deps, install: true)

