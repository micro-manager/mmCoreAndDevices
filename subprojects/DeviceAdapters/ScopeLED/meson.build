

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ScopeLEDDevices.cpp',
	'USBCommAdapter.cpp'
]


shared_library('ScopeLED', sources, dependencies : deps, install: true)

