

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LeicaDMI.cpp',
	'LeicaDMIModel.cpp',
	'LeicaDMIScopeInterface.cpp'
]


shared_library('LeicaDMI', sources, include_directories: includes, dependencies : deps, install: true)

