

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LeicaDMSTC.cpp',
	'LeicaDMSTCHub.cpp'
]


library('LeicaDMSTC', sources, include_directories: includes, dependencies : deps, install: true)

