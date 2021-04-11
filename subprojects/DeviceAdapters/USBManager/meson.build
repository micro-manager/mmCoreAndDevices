

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'USBManager.cpp'
]


shared_library('USBManager', sources, include_directories: includes, dependencies : deps, install: true)

