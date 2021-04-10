

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Mightex_USBCamera.cpp'
]


shared_library('Mightex_C_Cam', sources, include_directories: includes, dependencies : deps, install: true)

