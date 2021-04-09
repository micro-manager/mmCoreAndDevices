

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CRCChecksum.cpp',
	'OVP_ECS2.cpp'
]


shared_library('OVP_ECS2', sources, dependencies : deps, install: true)

