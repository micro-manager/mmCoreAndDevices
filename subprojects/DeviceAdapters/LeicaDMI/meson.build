

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LeicaDMI.cpp',
	'LeicaDMIModel.cpp',
	'LeicaDMIScopeInterface.cpp'
]


shared_library('LeicaDMI', sources, dependencies : deps, install: true)

