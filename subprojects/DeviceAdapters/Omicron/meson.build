

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CoherentOBISDirect.cpp',
	'Omicron.cpp',
	'OmicronDeviceDriver.cpp',
	'OmicronxX.cpp'
]


shared_library('Omicron', sources, dependencies : deps, install: true)

