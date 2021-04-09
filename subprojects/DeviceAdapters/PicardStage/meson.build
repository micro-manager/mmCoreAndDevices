

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PicardStage.cpp'
]


shared_library('PicardStage', sources, dependencies : deps, install: true)

