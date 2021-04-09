

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'TriggerScope.cpp'
]


shared_library('TriggerScope', sources, dependencies : deps, install: true)

