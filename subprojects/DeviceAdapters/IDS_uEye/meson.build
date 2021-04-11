

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'IDS_uEye.cpp'
]


shared_library('IDS_uEye', sources, include_directories: includes, dependencies : deps, install: true)

