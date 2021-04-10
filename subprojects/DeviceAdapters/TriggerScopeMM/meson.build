

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'TriggerScopeMM.cpp',
	'TriggerScopeMMDAC.cpp',
	'TriggerScopeMMTTL.cpp'
]


shared_library('TriggerScopeMM', sources, include_directories: includes, dependencies : deps, install: true)

