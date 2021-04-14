

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ModbusModule.cpp'
]


library('Modbus', sources, include_directories: includes, dependencies : deps, install: true)

