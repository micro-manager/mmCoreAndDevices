

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PicardStage.cpp'
]


shared_library('PicardStage', sources, include_directories: includes, dependencies : deps, install: true)

