

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FreeSerialPort.cpp'
]


shared_library('FreeSerialPort', sources, dependencies : deps, install: true)

