

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'XCiteViper.cpp'
]


shared_library('USB_Viper_QPL', sources, include_directories: includes, dependencies : deps, install: true)

