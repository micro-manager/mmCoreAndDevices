

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CanonEDSDK.cpp'
]


shared_library('CanonEDSDK', sources, include_directories: includes, dependencies : deps, install: true)

