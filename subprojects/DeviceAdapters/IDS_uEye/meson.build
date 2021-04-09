

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'IDS_uEye.cpp'
]


shared_library('IDS_uEye', sources, dependencies : deps, install: true)

