

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FreeSerialPort.cpp'
]


library('FreeSerialPort', sources, include_directories: includes, dependencies : deps, install: true)

