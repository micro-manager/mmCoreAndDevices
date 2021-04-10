

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LeicaDMSTC.cpp',
	'LeicaDMSTCHub.cpp'
]


shared_library('LeicaDMSTC', sources, include_directories: includes, dependencies : deps, install: true)

