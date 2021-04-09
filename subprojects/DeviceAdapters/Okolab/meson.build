

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'OkolabDevice.cpp'
]


shared_library('Okolab', sources, dependencies : deps, install: true)

